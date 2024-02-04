# %% [markdown]
# # CldNet模型数据加载
# 
# 动态加载文件名再根据文件名导入数据
# 

# %% [markdown]
# ## 数据加载

# %%
"""
Created on Wed June 14 16:18:10 2022
封装数据加载过程
@author: BEOH
"""
import itertools
import numpy as np
import xarray as xr


def load_data(stage: str = "train", **kwargs):
    """构造数据集
    kwargs:
        - stage: 'train'; if no obj => [inference]
        - RS_file: 遥感数据文件
        - Cld_file: 辐照数据文件
        - RS_Threshold: 遥感阈值
        - vr_file: 值区间
        - earth_relief_file: 海拔文件
        - earth_mask_file: 地球掩码文件
        - Bidx: 遥感通道组合
        - add_vars: 添加额外地理知识,可选["SAZ", "SAA", "SOZ", "SOA", "latitude", "longitude", "attitude", "earth_mask"]
        - RS_Vars: 默认即可
        - Cld_Vars: 默认即可
        - logger: optional 记录器
        - logNum: optional 记录器等级
        - is_norm: 是否标准化
    """
    if "RS_file" not in kwargs:
        kwargs["RS_file"] = None
        # raise ValueError("没有指定遥感数据")
    if "Cld_file" not in kwargs:
        kwargs["Cld_file"] = None
        # raise ValueError("没有指定辐照数据")
    if "RS_Threshold" not in kwargs:
        kwargs["RS_Threshold"] = None
        # raise ValueError("没有指定遥感阈值")
    if "vr_file" not in kwargs:
        kwargs["vr_file"] = None
    if "Bidx" not in kwargs:
        kwargs["Bidx"] = range(16)
    if "add_vars" not in kwargs:
        kwargs["add_vars"] = None
    if "RS_Vars" not in kwargs:
        kwargs["RS_Vars"] = None
    if "Cld_Vars" not in kwargs:
        kwargs["Cld_Vars"] = None
    if "logger" not in kwargs:
        kwargs["logger"] = None
    if "logNum" not in kwargs:
        kwargs["logNum"] = 10
    if "is_norm" not in kwargs:
        kwargs["is_norm"] = False
    elif kwargs["is_norm"]:
        if "vr_file" not in kwargs:
            raise ValueError("正则化但是没有给定value_range文件")

    if kwargs["RS_Vars"] is None:
        RS_Vars = [
            "albedo_01",
            "albedo_02",
            "albedo_03",
            "albedo_04",
            "albedo_05",
            "albedo_06",
            "tbb_07",
            "tbb_08",
            "tbb_09",
            "tbb_10",
            "tbb_11",
            "tbb_12",
            "tbb_13",
            "tbb_14",
            "tbb_15",
            "tbb_16",
        ]
    else:
        RS_Vars = kwargs["RS_Vars"]
    if kwargs["Cld_Vars"] is None:
        Cld_Vars = [
            "CLTYPE",
            "CLOT",
            "CLTT",
            "CLTH",
            "CLER_23",
        ]  # ['CLTYPE', 'CLOT', 'CLTT', 'CLTH', 'CLER_23']
    else:
        Cld_Vars = kwargs["Cld_Vars"]
    Combine_var = (
        list(itertools.combinations(range(1, 18), 1))
        + list(itertools.combinations(range(1, 7), 2))
        + list(itertools.combinations(range(7, 17), 2))
    )
    RS_Threshold = kwargs["RS_Threshold"]
    Bidx = kwargs["Bidx"]
    logger = kwargs["logger"]
    logNum = kwargs["logNum"]

    results = dict()
    if kwargs["RS_file"] is not None:
        data = xr.open_dataset(kwargs["RS_file"])
        if kwargs["RS_Threshold"] is not None:
            RS_temp1 = []
            for RS_Var in RS_Vars:
                RS_temp0 = data.variables[RS_Var].values
                RS_temp0[RS_temp0 < RS_Threshold[RS_Var][0]] = RS_Threshold[RS_Var][0]
                RS_temp0[RS_temp0 > RS_Threshold[RS_Var][1]] = RS_Threshold[RS_Var][1]
                RS_temp1.append(RS_temp0)
            RS_temp1 = np.stack(RS_temp1, axis=0)
        else:
            RS_temp1 = np.stack(
                [data.variables[RS_Var].values for RS_Var in RS_Vars], axis=0
            )
        RS_Data = []
        for ii in Bidx:
            try:
                if len(Combine_var[ii]) > 1:
                    idx0 = Combine_var[ii][0] - 1
                    idx1 = Combine_var[ii][1] - 1
                    RS_Data.append(RS_temp1[idx0] - RS_temp1[idx1])
                else:
                    idx0 = Combine_var[ii][0] - 1
                    RS_Data.append(RS_temp1[idx0])
            except Exception as e:
                print(f"Computational failed: {e}")
        RS_Data = np.stack(RS_Data, axis=0)
        if kwargs["is_norm"]:
            import pandas as pd

            vr = pd.read_csv(kwargs["vr_file"], index_col=0)
        if kwargs["is_norm"]:
            for ii in range(len(Bidx)):
                xname = Combine_var[Bidx[ii]]
                xname = str(xname)
                xmin = vr[xname]["min"]
                xmax = vr[xname]["max"]
                RS_Data[ii] = (RS_Data[ii] - xmin) / (xmax - xmin)
                if logger is not None:
                    logger.log(logNum, xname)

        if kwargs["add_vars"] is not None:
            add_data = []
            for xname in kwargs["add_vars"]:
                if xname in ["SAZ", "SAA", "SOZ", "SOA"]:
                    add_temp0 = data.variables[xname].values
                    if kwargs["is_norm"]:
                        xmin = vr[xname]["min"]
                        xmax = vr[xname]["max"]
                        add_temp0 = (add_temp0 - xmin) / (xmax - xmin)
                else:
                    continue
                add_data.append(add_temp0)
            if len(add_data) > 0:
                add_data = np.stack(add_data, axis=0)
                RS_Data = np.concatenate([RS_Data, add_data], axis=0)
        results["RS"] = RS_Data.astype("float32")

    if stage != "inference":
        if kwargs["Cld_file"] is not None:
            data = xr.open_dataset(kwargs["Cld_file"])
            for xname in Cld_Vars:
                results[xname] = data.variables[xname].values
                results[xname][results[xname] < 0.0] = np.nan
                if kwargs["is_norm"]:
                    xmin = vr[xname]["min"]
                    xmax = vr[xname]["max"]
                    if xname != "CLTYPE":
                        results[xname] = (results[xname] - xmin) / (xmax - xmin)

    if logger is not None:
        logger.log(logNum, f'{results["RS"].shape}, {results["RS"].dtype}')
        logger.info("Load data successfully!")

    return results

# %% [markdown]
# ## 构造数据类

# %%
"""
Created on Sun March 12 19:36:10 2023
封装加载的数据
@author: BEOH
"""
import torch

# 定义一个数据类


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        params,
        mask_ratio: float = -1.0,
        mask_bands=None,
        mask_delete: bool = False,
        resolution: float = 0.05,
        stage: str = "train",
        hw=[5, 5],
    ):
        """
        :params params: load_data函数的相关参数
        :params mask_ratio: 默认-1.0
        :params mask_bands: 默认None,即[]
        :params mask_delete: 默认False
        :params resolution: 默认0.05
        :params stage: 默认'train'; if no obj => [inference]
        """
        self.params = params.copy()
        self.mask_ratio = mask_ratio
        self.mask_bands = [] if mask_bands is None else mask_bands
        self.mask_delete = mask_delete
        self.len = len(self.params["RS_files"])
        self.stage = stage
        self.hw = hw
        res_num = int(120.0 / resolution) + 1

        if "Cld_Vars" not in self.params:
            self.params["Cld_Vars"] = ["CLTYPE", "CLOT", "CLTT", "CLTH", "CLER_23"]

        self.geoinfo = None
        if self.params["is_norm"]:
            import pandas as pd

            vr = pd.read_csv(self.params["vr_file"], index_col=0)
        if "add_vars" in self.params and self.params["add_vars"] is not None:
            add_data = []
            for xname in self.params["add_vars"]:
                if xname == "latitude":
                    add_temp0 = np.tile(np.linspace(60, -60, res_num), (res_num, 1)).T
                    if self.params["is_norm"]:
                        xmin = vr[xname]["min"]
                        xmax = vr[xname]["max"]
                        add_temp0 = (add_temp0 - xmin) / (xmax - xmin)
                elif xname == "longitude":
                    add_temp0 = np.tile(np.linspace(80, 200, res_num), (res_num, 1))
                    if self.params["is_norm"]:
                        xmin = vr[xname]["min"]
                        xmax = vr[xname]["max"]
                        add_temp0 = (add_temp0 - xmin) / (xmax - xmin)
                elif xname == "attitude":
                    add_temp0 = np.loadtxt(self.params["earth_relief_file"])
                    add_temp0[add_temp0 < 0] = 0.0
                    if self.params["is_norm"]:
                        xmin = vr[xname]["min"]
                        xmax = vr[xname]["max"]
                        add_temp0 = (add_temp0 - xmin) / (xmax - xmin)
                elif xname == "earth_mask":
                    add_temp0 = np.loadtxt(self.params["earth_mask_file"])
                    add_temp0[add_temp0 == 2] = 0.0
                    add_temp0[add_temp0 == 3] = 1.0
                    add_temp0[add_temp0 == 4] = 0.0
                    add_temp0 = add_temp0.astype("float32")
                else:
                    continue
                add_data.append(add_temp0)
            if len(add_data) > 0:
                self.geoinfo = np.stack(add_data, axis=0).astype("float32")

    def __getitem__(self, idx):
        load_data_params = self.params.copy()
        load_data_params["RS_file"] = self.params["RS_files"][idx]
        if self.stage != "inference":
            load_data_params["Cld_file"] = self.params["Cld_files"][idx]
        out = load_data(stage=self.stage, **load_data_params)
        if self.mask_ratio > 0.0 and self.mask_ratio <= 1.0:
            if self.mask_delete:
                out["RS"] = np.delete(out["RS"], self.mask_bands, axis=0)
            else:
                mask_id = np.random.choice(
                    [True, False],
                    out["RS"].shape[-2:],
                    p=[self.mask_ratio, 1.0 - self.mask_ratio],
                )
                for mask_band in self.mask_bands:
                    out["RS"][mask_band, mask_id] = 0

        if "Bidx01" not in load_data_params or len(load_data_params["Bidx01"]) == 0:
            out01 = None
        else:
            out01 = load_data(
                stage="inference",
                RS_file=self.params["RS_files"][idx],
                RS_Threshold=load_data_params["RS_Threshold"],
                vr_file=load_data_params["vr_file"],
                Bidx=load_data_params["Bidx01"],
            )

        if self.stage != "inference":
            if self.geoinfo is None:
                if out01 is not None:
                    return (
                        [out["RS"][:, :-1, :-1]]
                        + [out01["RS"][:, :-1, :-1]]
                        + [out[xx][:-1, :-1] for xx in self.params["Cld_Vars"]]
                    )
                else:
                    return [out["RS"][:, :-1, :-1]] + [
                        out[xx][:-1, :-1] for xx in self.params["Cld_Vars"]
                    ]
            else:
                if out01 is not None:
                    return (
                        [np.concatenate([out["RS"], self.geoinfo], axis=0)[:, :-1, :-1]]
                        + [out01["RS"][:, :-1, :-1]]
                        + [out[xx][:-1, :-1] for xx in self.params["Cld_Vars"]]
                    )
                else:
                    return [
                        np.concatenate([out["RS"], self.geoinfo], axis=0)[:, :-1, :-1]
                    ] + [out[xx][:-1, :-1] for xx in self.params["Cld_Vars"]]
        else:
            if self.geoinfo is None:
                if out01 is not None:
                    return out["RS"][:, :-1, :-1] + [out01["RS"][:, :-1, :-1]]
                else:
                    return out["RS"][:, :-1, :-1]
            else:
                if out01 is not None:
                    return np.concatenate([out["RS"], self.geoinfo], axis=0)[
                        :, :-1, :-1
                    ] + [out01["RS"][:, :-1, :-1]]
                else:
                    return np.concatenate([out["RS"], self.geoinfo], axis=0)[
                        :, :-1, :-1
                    ]

    def __len__(self):
        return self.len


