# CldNet Version 2.0 (CldNetV2)
Retrieval of all-day cloud property based on satellite remote sensing data

## Tutorial video
<!-- <video width="600" height="360" controls>
    <source src="Tutorial video (20240816_214812).mp4" type="video/mp4">
</video> -->
**YouTube**

[![CldNetV2 Tutorial](/Tutorial%20video%20cover.png)](https://youtu.be/64-1bdD89Ak)

**哔哩哔哩**

[![CldNetV2 Tutorial](/Tutorial%20video%20cover.png)](https://www.bilibili.com/video/BV1QwspeTEDE)


## File description
| Filename/Dirname                     | Description                                               |
| :----------------------------------- | :-------------------------------------------------------- |
| models                               | The specific parameters of the network structure CldNetV2 |
| inputs                               | Remote sensing files required for model input             |
| outputs                              | Cloud property files predicted by CldNetV2                |
| CldNetV2_inference_optimum.py        | Cloud property inference function                         |
| CldNetV2_inference_optimum.jsonc     | Cloud property inference configuration                    |
| Tutorial video (20240816_214812).mp4 | Tutorial video for cloud property retrieval               |

## Usage instructions
- Step 1: Download remote sensing images from satellite Himawari-8/9

Visit the [JAXA Himawari Monitor P-Tree System](https://www.eorc.jaxa.jp/ptree/index.html) (https://www.eorc.jaxa.jp/ptree/index.html), review the usage instructions or help documentation available on the site, and download the required data to your computer.

- Step 2: Install Anaconda

Visit the [Anaconda](https://www.anaconda.com/download/success) website (https://www.anaconda.com/download/success) to download the version suitable for your computer, and install it locally.

- Step 3: Create a Python virtual environment

Enter the following command in the terminal PowerShell.
```sh
conda create -n CldNetV2 python=3.12
```

- Step 4: Install the Python libraries required for the successful operation of the cloud retrieval system within the virtual environment CldNetV2

Enter the following command in the terminal PowerShell.
```sh
conda activate CldNetV2


# Pytorch for GPU+CPU
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# # or Pytorch for CPU
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu


pip install xarray, netCDF4, json5, pandas, einops


pip install nnn-2.5.7.tar.gz
```

- Step 5: Run the cloud retrieval system in the configured environment CldNetV2 to obtain the cloud property files.

Enter the following command in the terminal PowerShell.
```sh
# # To switch to the directory of the current project PMD-CldNetV2
# cd PMD-CldNetV2-main

conda activate CldNetV2

python CldNetV2_inference_optimum.py
```


## Other
If you have any questions, feel free to contact me via email metdai@outlook.com .

🌎🌍🌏