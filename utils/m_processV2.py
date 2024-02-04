import numpy as np
import torch
import torch.nn as nn
import einops
import copy
import collections


def demon(
    model,
    dataloader,
    criterion=None,
    optimizer=None,
    epoch_idx=None,
    logger=None,
    log_level=10,
    log_batch_num=10,
    target_names=["x0"],
    hw=None,
    nbs=None,
    func_loss=None,
    func_acc=None,
    pre_save: bool = False,
    ref_save: bool = False,
    is_train: bool = False
):
    """
    input:
    - model: 模型
    - dataloader: 数据加载器
    - criterion: 评价器, default: None. [criterion0, criterion1]
    - optimizer: 优化器, default: None.
    - scheduler: 学习率调整器, default: None.
    - epoch_idx: 第几轮, default: None.
    - logger: 记录器, default: None.
    - log_level: 记录器级别, default: 10.
    - log_batch_num: 每num个batch输出一次到屏幕, default: 5.
    - target_names: 目标名字, default: ["x0"].
    - hw: bs划分为n份, default: None.
    - nbs: bs划分为n份, default: 1.
    - func_loss: 损失函数, default: None.
    - func_acc: 精度函数, default: None.
    - pre_save: 针对预测结果, default: False.
    - ref_save: 针对参考结果, default: False.
    - is_train: False\n
    ---
    output:
    - process_loss: 损失值
    - process_pre: 预测值
    - process_ref: 参考值
    - process_acc: 准确度\n
    ---
    @Created on Sat January 01 15:39:20 2022
    @模型训练
    @author: BEOH
    @email: beoh86@yeah.net
    """

    model_device = next(model.parameters()).device
    if is_train:
        model.train()   # 切换模型为训练模式
    else:
        model.eval()    # 非训练状态下切换模型为评估模型

    process_loss = []
    process_pre = collections.OrderedDict()
    process_ref = collections.OrderedDict()
    process_acc = []
    if pre_save:
        for target_name in target_names:
            process_pre[target_name] = []
    if ref_save:
        for target_name in target_names:
            process_ref[target_name] = []

    with torch.set_grad_enabled(is_train):
        for batch_idx, batch_data in enumerate(dataloader):
            out_txt = f"epoch_idx: {epoch_idx}, batch_idx: {batch_idx}"

            # batch_data一定是list或者tuple
            if not isinstance(batch_data, (list, tuple)):
                batch_data = (batch_data,)
            batch_data = list(batch_data)
            # 训练情况下，并且元素个数大于1(至少一个输入和一个目标)
            if is_train and (not isinstance(batch_data, (list, tuple)) or len(batch_data) < 2):
                raise ValueError(
                    "when train is true, batch_data must be (list, tuple) and len(batch_data) > 1"
                )
            # 计算得到batch需要输入元素的个数
            if is_train or ref_save or func_loss is not None or func_acc is not None:
                batch_data_k = len(batch_data)-len(target_names)
            else:
                batch_data_k = len(batch_data)

            # 分割
            if hw is not None:
                for ii in range(batch_data_k):
                    batch_data[ii] = einops.rearrange(
                        batch_data[ii],
                        "b c (h2 h) (w2 w) -> (h2 w2 b) c h w",
                        h2=hw[0],
                        w2=hw[1]
                    )
            for ii in range(batch_data_k):
                batch_data[ii] = torch.chunk(batch_data[ii], nbs)

            # 前向计算
            outs = collections.OrderedDict()
            for target_name in target_names:
                outs[target_name] = []
            for iii in range(nbs):
                inputs = collections.OrderedDict()
                for ii in range(batch_data_k):
                    ii_name = "x{0}".format(ii)
                    inputs[ii_name] = batch_data[ii][iii]
                    if not isinstance(model, nn.DataParallel):
                        inputs[ii_name] = inputs[ii_name].to(model_device)
                output = model(inputs)
                for target_name in target_names:
                    outs[target_name].append(output[target_name])
            for target_name in target_names:
                outs[target_name] = torch.concat(outs[target_name])

            # 反分割
            if hw is not None:
                for target_name in target_names:
                    outs[target_name] = einops.rearrange(
                        outs[target_name],
                        "(h2 w2 b) c h w -> b c (h2 h) (w2 w)",
                        h2=hw[0],
                        w2=hw[1]
                    )

            # 参考值
            refs = collections.OrderedDict()
            for ii in range(batch_data_k, len(batch_data)):
                target_name = target_names[ii-batch_data_k]
                refs[target_name] = batch_data[ii]

            if pre_save:
                for target_name in target_names:
                    process_pre[target_name].append(outs[target_name])
            if ref_save:
                for target_name in target_names:
                    process_ref[target_name].append(refs[target_name])

            # 损失值计算
            if criterion is not None and func_loss is not None and callable(func_loss):
                loss, loss_ = func_loss(outs, refs, criterion)
                process_loss.append(loss_)
                out_txt = out_txt + \
                    f", loss: {round(loss.tolist(), 8)}-({[round(xx, 8) for xx in loss_]})"

            # 训练过程反向传播
            if is_train:
                optimizer.zero_grad()       # 梯度清零
                loss.backward()             # 反向传播，计算每层参数的梯度值
                optimizer.step()            # 更新参数，根据设置好的学习率迭代一步

            # 精度计算部分
            if func_acc is not None and callable(func_acc):
                acc = func_acc(outs, refs)
                process_acc.append(acc)
                out_txt = out_txt+f", accuracy: {round(acc, 6)}"

            # 打印部分结果到屏幕上
            if batch_idx % log_batch_num == 0:
                if logger is not None:
                    logger.log(log_level, out_txt)

    return process_loss, process_pre, process_ref, process_acc

