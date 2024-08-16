# %% [markdown]
# # CldNet Version 2.0
#
# Optimal inference (sheme 03)

# %% [markdown]
# ## 导入相应库

# %%
import os
import shutil
import json5
import argparse
import glob
import numpy as np
import pandas as pd
import xarray as xr
import torch

# -------------------工 具 函 数-------------------
from utils.m_logger import m_logger
from utils.m_processV2 import demon
from utils.get_Cld2NcP import get_Cld2NcP

# %% [markdown]
# ## 加载参数

# %%
parser = argparse.ArgumentParser(description="config parameters")
parser.add_argument(
    "--config",
    type=str,
    default="./CldNetV2_inference_optimum.jsonc",
    help="Program running parameters",
)
args, unknown = parser.parse_known_args()
with open(args.config, "r+", encoding="utf-8") as fp:
    f_config = json5.load(fp)

# 创建输出文件夹
if os.path.exists(f_config["out_dir"]):
    if f_config["is_remove"]:  # 删除输出文件夹
        shutil.rmtree(f_config["out_dir"])
        print("clear outdir!")
        os.makedirs(f_config["out_dir"])
else:
    os.makedirs(f_config["out_dir"])

# -------------------定义logger-------------------
log_file = os.path.join(f_config["out_dir"], "CldNetV2_inference_optimum.log")
logger = m_logger(log_file, f_config["logging"])
# 储存模型参数
logger.info("Version of PyTorch: {0}".format(torch.__version__))
device = f_config["device"]
if device == "cuda":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} device")

# %% [markdown]
# ## 导入数据加载函数

# %%
from utils.load_dataV2 import Dataset

# %% [markdown]
# ## 云属性推理函数

# %%
def get_Cld(
    load_data_params_daytime,
    load_data_params_nighttime,
    model_daytime1,
    model_nighttime1,
    model_daytime2,
    model_nighttime2,
    sign,
    endpoint=False,
):
    dataset_daytime = Dataset(
        load_data_params_daytime,
        mask_bands=f_config["load_data"]["mask_bands"]
        if "mask_bands" in f_config["load_data"]
        else None,
        mask_ratio=-1.0,
        mask_delete=f_config["load_data"]["mask_delete"]
        if "mask_delete" in f_config["load_data"]
        else False,
        stage="inference",
    )
    dataLoader_daytime = torch.utils.data.DataLoader(
        dataset_daytime, batch_size=1, shuffle=False, drop_last=False, num_workers=0
    )

    dataset_nighttime = Dataset(
        load_data_params_nighttime,
        mask_bands=f_config["load_data"]["mask_bands"]
        if "mask_bands" in f_config["load_data"]
        else None,
        mask_ratio=1.0,
        mask_delete=f_config["load_data"]["mask_delete"]
        if "mask_delete" in f_config["load_data"]
        else False,
        stage="inference",
    )
    dataLoader_nighttime = torch.utils.data.DataLoader(
        dataset_nighttime, batch_size=1, shuffle=False, drop_last=False, num_workers=0
    )

    # -------------------推理获得结果-------------------
    output_daytime = {}
    _, output_daytime, _, _ = demon(
        model_daytime2,
        dataLoader_daytime,
        logger=logger,
        log_level=10,
        target_names=["x0"]
        if "target_name" not in f_config
        else f_config["target_name"],
        hw=[5, 5] if "hw" not in f_config else f_config["hw"],
        nbs=1,
        pre_save=True,
        is_train=False,
    )
    for kk in model_daytime1.keys():
        if kk == "CLTYPE":
            output_daytime[kk] = demon(
                model_daytime1[kk],
                dataLoader_daytime,
                logger=logger,
                log_level=10,
                target_names=["x0"],
                hw=[5, 5] if "hw" not in f_config else f_config["hw"],
                nbs=1,
                pre_save=True,
                is_train=False,
            )[1]["x0"]
        else:
            output_daytime[kk] = demon(
                model_daytime1[kk],
                dataLoader_daytime,
                logger=logger,
                log_level=10,
                target_names=[kk],
                hw=[5, 5] if "hw" not in f_config else f_config["hw"],
                nbs=1,
                pre_save=True,
                is_train=False,
            )[1][kk]

    output_nighttime = {}
    _, output_nighttime, _, _ = demon(
        model_nighttime2,
        dataLoader_nighttime,
        logger=logger,
        log_level=10,
        target_names=["x0"]
        if "target_name" not in f_config
        else f_config["target_name"],
        hw=[5, 5] if "hw" not in f_config else f_config["hw"],
        nbs=1,
        pre_save=True,
        is_train=False,
    )
    for kk in model_nighttime1.keys():
        if kk == "CLTYPE":
            output_nighttime[kk] = demon(
                model_nighttime1[kk],
                dataLoader_nighttime,
                logger=logger,
                log_level=10,
                target_names=["x0"],
                hw=[5, 5] if "hw" not in f_config else f_config["hw"],
                nbs=1,
                pre_save=True,
                is_train=False,
            )[1]["x0"]

        else:
            output_nighttime[kk] = demon(
                model_nighttime1[kk],
                dataLoader_nighttime,
                logger=logger,
                log_level=10,
                target_names=[kk],
                hw=[5, 5] if "hw" not in f_config else f_config["hw"],
                nbs=1,
                pre_save=True,
                is_train=False,
            )[1][kk]

    # 保存变量
    Cld = []
    Cld_encoding = dict()
    clearsky_sign = None
    for xname in f_config["target_name"]:
        # logger.info(f"{'-'*60+xname+'-'*60}")
        if xname == "CLTYPE":
            temp = (
                torch.argmax(torch.softmax(output_daytime[xname][0], dim=1), dim=1)[0]
                .cpu()
                .numpy()
                .astype("float32")
            )
            clearsky_sign = temp == 0
        else:
            vr = pd.read_csv(f_config["load_data"]["params"]["vr_file"], index_col=0)
            xmin = vr[xname]["min"]
            xmax = vr[xname]["max"]
            temp = (
                output_daytime[xname][0][0, 0].cpu().numpy().astype("float32")
                * (xmax - xmin)
                + xmin
            )
            if clearsky_sign is not None:
                temp[clearsky_sign] = np.nan
            temp[temp <= 0] = np.nan
        temp_daytime = temp

        if xname == "CLTYPE":
            temp = (
                torch.argmax(torch.softmax(output_nighttime[xname][0], dim=1), dim=1)[0]
                .cpu()
                .numpy()
                .astype("float32")
            )
            clearsky_sign = temp == 0
        else:
            vr = pd.read_csv(f_config["load_data"]["params"]["vr_file"], index_col=0)
            xmin = vr[xname]["min"]
            xmax = vr[xname]["max"]
            temp = (
                output_nighttime[xname][0][0, 0].cpu().numpy().astype("float32")
                * (xmax - xmin)
                + xmin
            )
            if clearsky_sign is not None:
                temp[clearsky_sign] = np.nan
            temp[temp <= 0] = np.nan

        temp_nighttime = temp
        temp_daytime[sign] = temp_nighttime[sign]
        temp = temp_daytime
        if endpoint:
            temp = temp[list(range(temp.shape[0])) + [-1], :]
            temp = temp[:, list(range(temp.shape[1])) + [-1]]
            lon = np.linspace(80, 200, temp.shape[1], endpoint=endpoint)
            lat = np.linspace(60, -60, temp.shape[0], endpoint=endpoint)
        else:
            lon = np.linspace(80, 200, temp.shape[1], endpoint=endpoint)
            lat = np.linspace(60, -60, temp.shape[0], endpoint=endpoint)
        Cld.append(
            xr.DataArray(
                data=temp,
                dims=("latitude", "longitude"),
                coords={"latitude": lat, "longitude": lon},
                name=xname,
                attrs=get_Cld2NcP(xname, flag="attrs"),
            )
        )
        Cld_encoding[xname] = get_Cld2NcP(xname, flag="encoding")

    # add Using Model Identification (UMI)
    if endpoint:
        sign = sign[list(range(sign.shape[0])) + [-1], :]
        sign = sign[:, list(range(sign.shape[1])) + [-1]]
        lon = np.linspace(80, 200, sign.shape[1], endpoint=endpoint)
        lat = np.linspace(60, -60, sign.shape[0], endpoint=endpoint)
    else:
        lon = np.linspace(80, 200, sign.shape[1], endpoint=endpoint)
        lat = np.linspace(60, -60, sign.shape[0], endpoint=endpoint)
    Cld.append(
        xr.DataArray(
            data=sign.astype(int),
            dims=("latitude", "longitude"),
            coords={"latitude": lat, "longitude": lon},
            name="UMI",
            attrs=dict(
                long_name="Using Model Identification",
                units="Dimensionless",
                valid_min="0",
                valid_max="1",
            ),
        )
    )
    Cld_encoding["UMI"] = {
        "dtype": "int16",
        "scale_factor": 1.0,
        "add_offset": 0.0,
        "zlib": True,
        "_FillValue": 255,
    }

    Cld = xr.merge(Cld)
    return Cld, Cld_encoding

# %% [markdown]
# ## 推理过程

# %%
# 获取RS阈值
with open(
    f_config["load_data"]["params"]["RS_Threshold_file"], "r+", encoding="utf-8"
) as fp:
    RS_Threshold = json5.load(fp)
f_config["load_data"]["params"]["RS_Threshold"] = RS_Threshold

# 加载模型参数, 实例化模型
models1 = {}
models1["W"] = {}
for kk, vv in f_config["checkpoint1"]["W"].items():
    models1["W"][kk] = torch.load(vv).to(device)
models1["O"] = {}
for kk, vv in f_config["checkpoint1"]["O"].items():
    models1["O"][kk] = torch.load(vv).to(device)
models2 = {}
models2["W"] = torch.load(f_config["checkpoint2"]["W"]).to(device)
models2["O"] = torch.load(f_config["checkpoint2"]["O"]).to(device)

# 生成RS文件列表
# NC_H08_20221230_0300_R21_FLDK.02401_02401.nc
# NC_H08_20221230_0300_L2CLP010_FLDK.02401_02401.nc
RS_dir = f_config["load_data"]["params"]["RS_dir"]
RS_files = glob.glob(RS_dir + "NC_H08_*_R21_FLDK.02401_02401.nc")
logger.info(f"{'-'*30}Inference starts{'-'*30}")
for RS_file in RS_files:
    # -------------------构建输出目标文件名-------------------
    Cld_file = (
        f_config["out_dir"]
        + os.path.basename(RS_file)[:21]
        + "L2CLP010_FLDK.02401_02401.nc"
    )
    sign = xr.open_dataset(RS_file).variables["SOZ"][:-1, :-1].values > f_config["SOZ"]

    # -------------------构建数据加载器-------------------
    load_data_params = f_config["load_data"]["params"].copy()
    load_data_params["RS_files"] = [RS_file]
    if "add_vars" in load_data_params and isinstance(
        load_data_params["add_vars"], (dict,)
    ):
        if "W" in load_data_params["add_vars"].keys():
            load_data_params["add_vars"] = load_data_params["add_vars"]["W"]
        else:
            load_data_params.pop("add_vars")
    load_data_params_daytime = load_data_params

    load_data_params = f_config["load_data"]["params"].copy()
    load_data_params["RS_files"] = [RS_file]
    if "add_vars" in load_data_params and isinstance(
        load_data_params["add_vars"], (dict,)
    ):
        if "O" in load_data_params["add_vars"].keys():
            load_data_params["add_vars"] = load_data_params["add_vars"]["O"]
        else:
            load_data_params.pop("add_vars")
    load_data_params_nighttime = load_data_params

    # ---------------------云属性推理---------------------
    Cld, Cld_encoding = get_Cld(
        load_data_params_daytime,
        load_data_params_nighttime,
        models1["W"],
        models1["O"],
        models2["W"],
        models2["O"],
        sign,
        endpoint=False if "endpoint" not in f_config else f_config["endpoint"],
    )
    Cld.attrs = get_Cld2NcP("Cloud", flag="attrs")
    Cld.attrs["SOZ_threshold"] = f_config["SOZ"]
    Cld.attrs[
        "UMI"
    ] = "Using Model Identification (0 represents using the model CldNet-W, and 1 represents using the model CldNet-O.)"
    Cld_encoding = {
        "CLTYPE": {"zlib": True, "_FillValue": 255},
        "CLOT": {"zlib": True, "_FillValue": -9999},
        "CLTT": {"zlib": True, "_FillValue": -9999},
        "CLTH": {"zlib": True, "_FillValue": -9999},
        "CLER_23": {"zlib": True, "_FillValue": -9999},
        "UMI": {"zlib": True, "_FillValue": 255},
    }
    Cld.to_netcdf(Cld_file, encoding=Cld_encoding)
    logger.info(f"{'-'*30}{os.path.basename(RS_file)[:20]}{'-'*30}")
logger.info(f"Inference completed successfully......")
