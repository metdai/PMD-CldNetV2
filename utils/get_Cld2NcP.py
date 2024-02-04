# %% [markdown]
# # 保存NC设置属性

# %%
import datetime


def get_Cld2NcP(xname='CLTYPE', flag="encoding"):
    """
    input:
    - xname: 云类型名, default: 'CLTYPE'. Options: ['CLTYPE', 'CLOT', 'CLTT', 'CLTH', 'CLER_23']
    - flag: 标记, default: 'encoding'. Options: ['encoding', 'attrs']\n
    ---
    output:
    ---
    @Created on Sat January 01 15:39:20 2022
    @模型训练
    @author: BEOH
    @email: beoh86@yeah.net
    """

    if xname == "CLTYPE":
        if flag == 'attrs':
            return dict(
                long_name="Cloud Type under ISCCP Cloud Type Classification Definition",
                units="Dimensionless",
                valid_min="0",
                valid_max="9",
                description="0=Clear, 1=Ci, 2=Cs, 3=Deep convection, 4=Ac, 5=As, 6=Ns, 7=Cu, 8=Sc, 9=St",
            )
        elif flag == 'encoding':
            return {"dtype": "int16", "scale_factor": 1.0, "add_offset": 0.0, "zlib": True, '_FillValue': 255}
    elif xname == "CLOT":
        if flag == 'attrs':
            return dict(
                long_name="Cloud Optical Thickness",
                units="Dimensionless",
                valid_min="0",
                valid_max="150"
            )
        elif flag == 'encoding':
            return {"dtype": "int16", "scale_factor": 0.01, "add_offset": 0.0, "zlib": True, '_FillValue': -9999}
    elif xname == "CLTT":
        if flag == 'attrs':
            return dict(
                long_name="Cloud Top Temperature",
                units="K",
                valid_min="0",
                valid_max="350"
            )
        elif flag == 'encoding':
            return {"dtype": "int16", "scale_factor": 0.01, "add_offset": 150.0, "zlib": True, '_FillValue': -9999}
    elif xname == "CLTH":
        if flag == 'attrs':
            return dict(
                long_name="Cloud Top Height",
                units="km",
                valid_min="0",
                valid_max="20"
            )
        elif flag == 'encoding':
            return {"dtype": "int16", "scale_factor": 0.001, "add_offset": 0.0, "zlib": True, '_FillValue': -9999}
    elif xname == "CLER_23":
        if flag == 'attrs':
            return dict(
                long_name="Cloud Effective Radius",
                units="micro meter",
                valid_min="0",
                valid_max="100"
            )
        elif flag == 'encoding':
            return {"dtype": "int16", "scale_factor": 0.01, "add_offset": 0.0, "zlib": True, '_FillValue': -9999}
    elif xname == "Cloud":
        if flag == 'attrs':
            return dict(
                title="All-weather cloud retrieval",
                keywords="Himawari8/9, CldNet, KBDD",
                creator_name="Beoh",
                email="beoh86@yeah.net",
                date_created=str(datetime.datetime.utcnow()),
                upper_left_latitude=60.0,
                upper_left_longitude=80.0,
                grid_interval=0.05
            )
        elif flag == 'encoding':
            return {}
    else:
        raise ValueError("没有云属性值")



