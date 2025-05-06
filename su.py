import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import numpy as np


# Dataset class for SU3 dataset
class SU3Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_names = [f[:-4] for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        base_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, base_name + '.png')
        label_path = os.path.join(self.image_dir, base_name + '_label.npz')

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = np.load(label_path)['vpts']
        heatmaps = torch.tensor(labels, dtype=torch.float32)
        return image, heatmaps


# Relative Positional Attention Module
class RelativePositionalAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


# VP-SAT Model
class VPSAT(nn.Module):
    def __init__(self, num_heads=8, hidden_dim=256):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.backbone = create_feature_extractor(backbone, {'layer3': 'feat'})
        self.conv_proj = nn.Conv2d(1024, hidden_dim, kernel_size=1)

        self.rpa = RelativePositionalAttention(hidden_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4),
            num_layers=6
        )

        self.fc_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, 9)  # Predict 3 VPs as (x, y, z)
        )

    def forward(self, x):
        feat = self.backbone(x)['feat']
        feat = self.conv_proj(feat)
        B, C, H, W = feat.shape

        feat_flat = feat.flatten(2).permute(0, 2, 1)
        feat_rpa = self.rpa(feat_flat)

        encoded = self.transformer_encoder(feat_rpa)
        encoded_feat = encoded.permute(0, 2, 1).reshape(B, C, H, W)

        vps = self.fc_head(encoded_feat)
        vps = vps.view(-1, 3, 3)
        return vps / vps.norm(dim=-1, keepdim=True)


def angular_accuracy(preds, gts, thresholds=[1, 2, 5]):
    def angle_diff(v1, v2):
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_angle) * (180.0 / np.pi)
        return angle

    accuracies = np.zeros(len(thresholds))
    total = preds.shape[0] * preds.shape[1]

    for pred, gt in zip(preds, gts):
        for p, g in zip(pred, gt):
            angle = angle_diff(p, g)
            for i, thresh in enumerate(thresholds):
                if angle <= thresh:
                    accuracies[i] += 1

    return accuracies / total


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VPSAT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    criterion = nn.MSELoss()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    train_dataset, test_dataset = random_split(SU3Dataset(image_dir='data/', transform=transform), lengths=[0.9, 0.1])
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model.train()

    losses = []
    for epoch in range(0):
        epoch_loss = 0
        for images, gt_vps in train_loader:
            images, gt_vps = images.to(device), gt_vps.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, gt_vps)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')
    # Plot loss curve
    # plt.plot(range(1, 11), losses, marker='o')
    # plt.title('Training Loss Curve')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.show()

    # Evaluation
    model.eval()
    acc = np.array([0,0,0])
    for images, gt_vps in test_loader:
        images, gt_vps = images.to(device), gt_vps
        pred_vps = model(images).cpu().detach().numpy()
        gt_vps = gt_vps.numpy()
        acc = angular_accuracy(pred_vps, gt_vps)



    # Predict

    sample_image, gt_vps = train_dataset[0]
    input_img = sample_image.unsqueeze(0).to(device)
    pred_vps = model(input_img).cpu().detach().numpy()
    gt_vps = gt_vps.numpy().reshape(1, 3, 3)

    acc = angular_accuracy(pred_vps, gt_vps)
    print("Angular accuracy @1°, 2°, 5°:", acc)

    # Visualize predictions
    plt.imshow(sample_image.permute(1, 2, 0).numpy())
    plt.title('Sample Image with Predicted VPs')
    plt.axis('off')

    h, w = 512, 512
    for vp in pred_vps:
        vp = vp / np.abs(vp[-1])
        x = list((vp[0] * w / 2) + w / 2)
        y = list((vp[1] * h / 2) + h / 2)

        # x, y = int((vp[0] * w / 2) + w / 2), int((vp[1] * h / 2) + h / 2)
        plt.plot(x, y, 'ro')

    plt.show()
