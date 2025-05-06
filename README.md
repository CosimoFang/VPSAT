# VPSAT
### Downloading the Processed Datasets

Make sure `curl` is installed on your system and execute

```bash
wget https://huggingface.co/yichaozhou/neurvps/resolve/main/Data/su3.tar.xz
tar xf su3.tar.xz
rm *.tar.xz *.z*
cd ..
```
run process.py to process the su3 data

run su.py to train the model and do the evaluation
