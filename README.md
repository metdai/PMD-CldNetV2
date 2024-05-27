# CldNet Version 2.0 (CldNetV2)
Retrieval of all-day cloud property based on satellite remote sensing data

## File description
| Filename/Dirname                 | Description                                               |
| :------------------------------- | :-------------------------------------------------------- |
| models                           | The specific parameters of the network structure CldNetV2 |
| inputs                           | Remote sensing files required for model input             |
| outputs                          | Cloud property files predicted by CldNetV2                |
| CldNetV2_inference_optimum.py    | Cloud property inference function                         |
| CldNetV2_inference_optimum.jsonc | Cloud property inference configuration                    |

## Inference 🚀
- Firstly, create a Python virtual environment.
```sh
conda create -n CldNetV2 python=3.12
```

- Secondly, activate the virtual environment and install related packages.
```sh
conda activate CldNetV2
conda create -n CldNetV2 python=3.12
conda activate CldNetV2
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install xarray, netCDF4, json5, pandas, einops
pip install nnn-2.5.7.tar.gz
```

- Finally, run the following command to predict cloud property in in the terminal.
```sh
python CldNetV2_inference_optimum.py
```

## Other
If you have any questions, feel free to contact me via email metdai@outlook.com .

🌎🌍🌏