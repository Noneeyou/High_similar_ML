#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KAIST 数据集处理工具
====================

功能：
- 处理 KAIST 提供的四类传感器数据（振动、声学、温度、电流）
- 输入原始文件路径（.mat / .tdms），输出统一格式的 CSV 文件
- 输出文件与原文件同名，存放在指定目录下

依赖：
    pip install scipy pandas nptdms
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from nptdms import TdmsFile


# -------------------
# VIBRATION (.mat)
# -------------------
def process_vibration(file_path: str, save_dir: str) -> str:
    """
    处理 KAIST 振动数据 .mat 文件 (Signal 结构)，保存为 CSV。
    """
    data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    signal = data["Signal"]

    start = float(signal.x_values.start_value)
    inc = float(signal.x_values.increment)
    n = int(signal.x_values.number_of_values)
    time = start + np.arange(n) * inc

    values = np.array(signal.y_values.values)

    if values.ndim == 1:
        values = values[:, None]

    df = pd.DataFrame(
        values,
        columns=[f"vibration_ch{i+1}" for i in range(values.shape[1])]
    )
    df.insert(0, "timestamp", time)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        os.path.basename(file_path).replace(".mat", ".csv")
    )
    df.to_csv(save_path, index=False)
    return save_path


# -------------------
# ACOUSTIC (.mat)
# -------------------
def process_acoustic(file_path: str, save_dir: str) -> str:
    """
    处理 KAIST 声学数据 .mat 文件 (Signal 结构)，保存为 CSV。
    """
    data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    signal = data["Signal"]

    start = float(signal.x_values.start_value)
    inc = float(signal.x_values.increment)
    n = int(signal.x_values.number_of_values)
    time = start + np.arange(n) * inc

    values = np.array(signal.y_values.values).reshape(-1)

    df = pd.DataFrame({
        "timestamp": time,
        "acoustic": values
    })

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        os.path.basename(file_path).replace(".mat", ".csv")
    )
    df.to_csv(save_path, index=False)
    return save_path


# -------------------
# TEMPERATURE + CURRENT (.tdms)
# -------------------
def process_tdms(file_path: str, save_dir: str) -> str:
    """
    处理 KAIST 温度 + 电流数据 .tdms 文件，保存为 CSV。

    输出列名：
        timestamp,
        Temp_A(°C), Temp_B(°C),
        I_U(A), I_V(A), I_W(A)

    说明：
    - TDMS 各通道采样率不同，长度可能不一致
    - 本函数采用「截断到最短通道长度」策略进行对齐
    """

    tdms_file = TdmsFile.read(file_path)

    # 读取所有通道
    data_dict = {}
    for group in tdms_file.groups():
        for ch in group.channels():
            name = ch.name.strip()
            data_dict[name] = np.asarray(ch[:]).reshape(-1)

    if not data_dict:
        raise RuntimeError(f"TDMS 文件 {file_path} 中未读取到任何通道")

    # 提取时间轴
    time = None
    for key in ["Time Stamp", "Time_Stamp", "timestamp", "Time"]:
        if key in data_dict:
            time = data_dict.pop(key)
            break

    if time is None:
        first_len = len(next(iter(data_dict.values())))
        time = np.arange(first_len, dtype=float)

    # 通道重命名
    rename_map = {
        "Temperature_housing_A": "Temp_A(°C)",
        "Temperature_housing_B": "Temp_B(°C)",
        "U-phase": "I_U(A)",
        "V-phase": "I_V(A)",
        "W-phase": "I_W(A)"
    }

    renamed = {
        rename_map.get(k, k): v
        for k, v in data_dict.items()
    }

    # ===============================
    # 关键修复：对齐所有通道长度
    # ===============================
    min_len = min(len(v) for v in renamed.values())
    min_len = min(min_len, len(time))

    aligned = {
        k: v[:min_len]
        for k, v in renamed.items()
    }

    df = pd.DataFrame(aligned)
    df.insert(0, "timestamp", time[:min_len])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        os.path.basename(file_path).replace(".tdms", ".csv")
    )
    df.to_csv(save_path, index=False)
    return save_path


# -------------------
# 总调度函数
# -------------------
def process_kaist_file(file_path: str, save_dir: str, data_type: str) -> str:
    """
    统一入口：根据数据类型选择对应分支处理

    参数:
        file_path : str
            原始文件路径
        save_dir : str
            存放目录
        data_type : str
            {"vibration", "acoustic", "temperature", "current", "temperature_current"}

    返回:
        str - 保存的 CSV 文件路径
    """
    if data_type == "vibration":
        return process_vibration(file_path, save_dir)
    elif data_type == "acoustic":
        return process_acoustic(file_path, save_dir)
    elif data_type in ("temperature", "current", "temperature_current"):
        return process_tdms(file_path, save_dir)
    else:
        raise ValueError(f"未知的数据类型: {data_type}")
