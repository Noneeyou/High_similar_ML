import os
import scipy.io as sio
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

def mat_to_csv(mat_file, output_dir):
    """
    将 MAT 文件转换为两个 CSV 文件：
    1. <原文件名>_data.csv: 时间轴与信号值一一对应
    2. <原文件名>_metadata.csv: function_record 内的说明信息
    
    参数:
        mat_file: str, 输入的 .mat 文件路径
        output_dir: str, 输出文件夹路径
    """
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(mat_file))[0]
    data_csv = os.path.join(output_dir, f"{base_name}_data.csv")
    meta_csv = os.path.join(output_dir, f"{base_name}_metadata.csv")

    # 检查文件是否存在
    for f in [data_csv, meta_csv]:
        if os.path.exists(f):
            choice = input(f"文件 {f} 已存在，是否覆盖？(y/n): ").strip().lower()
            if choice != "y":
                print(f"跳过保存 {f}")
                return None, None

    # 读取 mat 文件
    mat_data = sio.loadmat(mat_file)
    signal = mat_data['Signal'][0]  # 取第一个信号对象

    # ========== 数据部分 ==========
    x_values = signal['x_values'][0]
    y_values = signal['y_values'][0]

    start_value = float(x_values['start_value'][0][0])
    increment = float(x_values['increment'][0][0])
    number_of_values = int(x_values['number_of_values'][0][0])

    # 时间序列
    time_values = np.arange(start_value, start_value + number_of_values * increment, increment)
    time_values = time_values[:number_of_values]

    # 信号值
    signal_data = y_values['values'][0][0].flatten()

    # 保存 data.csv
    df = pd.DataFrame({"Time": time_values, "Signal": signal_data})
    df.to_csv(data_csv, index=False)

    # ========== 元数据部分 ==========
    function_record = signal['function_record'][0]
    meta_dict = {}
    for name in function_record.dtype.names:
        try:
            val = function_record[name][0][0]
            if isinstance(val, np.ndarray) and val.size == 1:
                val = val.item()
            meta_dict[name] = val
        except Exception as e:
            meta_dict[name] = str(function_record[name])

    meta_df = pd.DataFrame(list(meta_dict.items()), columns=["Field", "Value"])
    meta_df.to_csv(meta_csv, index=False)

    print(f"数据已保存到:\n {data_csv}\n {meta_csv}")
    return data_csv, meta_csv

import re
from typing import List

__all__ = ["tdms_to_csv"]  # 仅暴露公共函数


def _safe_name(s: str) -> str:
    """
    私有：将组名/通道名转换为安全的文件名片段
    - 非字母数字字符替换为下划线
    - 去掉前后多余下划线
    """
    return re.sub(r"[^\w\-]+", "_", str(s)).strip("_")


def _build_time_axis_from_props(channel) -> np.ndarray | None:
    """
    私有：根据 TDMS 通道属性尝试构造时间轴
    优先使用 channel.time_track()；否则根据 wf_increment(+样本数) 构造相对时间
    返回:
        np.ndarray | None
    """
    try:
        tt = channel.time_track()  # nptdms 新版本提供
        if tt is not None:
            return np.asarray(tt)
    except Exception:
        pass

    props = getattr(channel, "properties", {}) or {}
    wf_inc = props.get("wf_increment", None)
    if wf_inc is not None:
        n = props.get("wf_samples", None)
        if n is None:
            try:
                n = len(channel)
            except Exception:
                n = None
        if n is not None:
            return np.arange(n, dtype=float) * float(wf_inc)
    return None


def tdms_to_csv(tdms_file: str, output_dir: str) -> str:
    """
    将 .tdms 文件展开并保存到一个 CSV 文件中

    参数:
        tdms_file (str): 输入 .tdms 文件路径
        output_dir (str): 导出 .csv 文件保存目录

    返回:
        str: 最终保存的 CSV 文件路径

    依赖:
        pip install nptdms

    说明:
        - 输出文件名与输入文件一致，仅后缀改为 .csv
        - CSV 中包含: group, channel, time(可选), value
    """
    from nptdms import TdmsFile  # 延迟导入

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成输出文件路径
    base_name = os.path.splitext(os.path.basename(tdms_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.csv")

    tdms = TdmsFile.read(tdms_file)

    rows = []

    # 遍历所有组与通道
    for group in tdms.groups():
        gname = _safe_name(group.name)
        for channel in group.channels():
            cname = _safe_name(channel.name)
            values = np.asarray(channel[:])

            # 生成时间轴
            t = _build_time_axis_from_props(channel)
            if t is not None and len(t) == len(values):
                for time, val in zip(t, values):
                    rows.append({"group": gname, "channel": cname,
                                 "time": time, "value": val})
            else:
                for val in values:
                    rows.append({"group": gname, "channel": cname,
                                 "time": None, "value": val})

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"✅ 已保存: {output_file}")
    return output_file


def build_local_temporal_graph(
    csv_path: str,
    save_dir: str,
    num_edges: int = 10,
    label_col: int = None
):
    """
    基于时间顺序构建局部时序图。
    每一行是一个节点，上下相邻样本构成边。
    标签列可指定索引，若不指定则默认最后一列。
    🚫 自动忽略首列（常用于序号/ID），避免误入特征计算。

    参数:
        csv_path (str): 输入 CSV 文件路径。
        save_dir (str): 图结构文件的保存文件夹。
        num_edges (int): 每个节点的边数（上下平均分配）。
        label_col (int): 标签列索引（默认 None → 最后一列）。
    返回:
        (nodes_csv, edges_csv, graph_pt): 保存的文件路径元组。
    """
    os.makedirs(save_dir, exist_ok=True)

    # === 读取数据 ===
    df = pd.read_csv(csv_path)
    num_nodes = len(df)
    if num_nodes == 0:
        raise ValueError("❌ 输入 CSV 文件为空。")

    # === 标签列判断 ===
    if label_col is None:
        label_col = df.shape[1] - 1

    # === 提取标签列 ===
    y = torch.tensor(df.iloc[:, label_col].values, dtype=torch.long)

    # === 构造特征列（去掉首列 + 标签列）===
    drop_cols = [df.columns[0], df.columns[label_col]] if label_col != 0 else [df.columns[0]]
    df_features = df.drop(columns=drop_cols, errors="ignore")

    # 保留数值列
    df_features = df_features.select_dtypes(include=["float", "int"])
    if df_features.shape[1] == 0:
        raise ValueError("❌ 特征列为空，请检查输入 CSV。")

    # === 构建边 ===
    half = num_edges // 2
    edges = []
    for i in range(num_nodes):
        start_up = max(0, i - half)
        end_down = min(num_nodes, i + half + 1)
        up_neighbors = list(range(start_up, i))
        down_neighbors = list(range(i + 1, end_down))

        total_needed = num_edges
        current = len(up_neighbors) + len(down_neighbors)
        if current < total_needed:
            remaining = total_needed - current
            if i + half + 1 >= num_nodes:  # 下方不够
                extra_up = list(range(max(0, start_up - remaining), start_up))
                up_neighbors = extra_up + up_neighbors
            elif i - half < 0:  # 上方不够
                extra_down = list(range(end_down, min(num_nodes, end_down + remaining)))
                down_neighbors += extra_down

        for j in up_neighbors + down_neighbors:
            edges.append((i, j))
            edges.append((j, i))

    # === 保存节点与边 ===
    nodes_path = os.path.join(save_dir, "nodes.csv")
    edges_path = os.path.join(save_dir, "edges.csv")
    graph_path = os.path.join(save_dir, "graph.pt")

    df_features.to_csv(nodes_path, index=False)
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(edges_path, index=False)

    # === 构建 PyG 图结构 ===
    edge_index = torch.tensor(edges, dtype=torch.long).T
    x = torch.tensor(df_features.values, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)

    torch.save(data, graph_path)

    print(f"✅ 图结构构建完成，共 {num_nodes} 个节点，{len(edges)//2} 条无向边")
    print(f"📁 nodes.csv: {nodes_path}")
    print(f"📁 edges.csv: {edges_path}")
    print(f"📁 graph.pt : {graph_path}")
    print(f"🧩 特征维度: {x.shape[1]} (已自动忽略首列与标签列)")

    return nodes_path, edges_path, graph_path


def build_similarity_knn_graph(
    csv_path: str,
    save_dir: str,
    num_edges: int = 10,
    label_col: int = None
):
    """
    基于样本间余弦相似度 + KNN 建图。
    可指定标签列索引；若不指定则默认最后一列。
    输出结构与 build_local_temporal_graph 一致。

    参数:
        csv_path (str): 输入 CSV 文件路径。
        save_dir (str): 图结构文件的保存文件夹。
        num_edges (int): 每个节点连接的邻点数(KNN数量)。
        label_col (int): 标签列索引（默认 None → 最后一列）。
    返回:
        (nodes_csv, edges_csv, graph_pt): 保存的文件路径元组。
    """

    # === 读取数据 ===
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 找不到输入文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"📊 已读取数据: {df.shape}")

    # === 提取标签列 ===
    if label_col is None:
        label_col = df.shape[1] - 1
    y = torch.tensor(df.iloc[:, label_col].values, dtype=torch.long)

    # 忽略首列（序号）+ 标签列
    df_features = df.drop(df.columns[[0, label_col]], axis=1, errors="ignore")
    df_features = df_features.select_dtypes(include=["float", "int"])

    features = df_features.values.astype(np.float32)
    num_nodes = features.shape[0]
    print(f"🧩 使用特征列数: {features.shape[1]} | 特征列示例: {list(df_features.columns)[:5]} ...")

    # === 计算余弦相似度 ===
    print("⚙️ 正在计算余弦相似度矩阵...")
    sim_matrix = cosine_similarity(features)
    np.fill_diagonal(sim_matrix, -np.inf)

    # === KNN 边构建 ===
    print(f"🔍 正在为每个节点选取 {num_edges} 个最相似邻居...")
    edges = []
    for i in range(num_nodes):
        topk_idx = np.argpartition(sim_matrix[i], -num_edges)[-num_edges:]
        for j in topk_idx:
            edges.append([i, j])
            edges.append([j, i])

    edges = np.array(edges)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)

    # === 构造 PyG 对象 ===
    data = Data(x=x, edge_index=edge_index, y=y)

    # === 保存 ===
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    nodes_csv = os.path.join(save_dir, f"{base_name}_nodes.csv")
    edges_csv = os.path.join(save_dir, f"{base_name}_edges.csv")
    graph_pt = os.path.join(save_dir, f"{base_name}_graph.pt")

    pd.DataFrame(features).to_csv(nodes_csv, index=False)
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(edges_csv, index=False)
    torch.save(data, graph_pt)

    print(f"✅ 图构建完成，共 {num_nodes} 个节点，{len(edges)//2} 条无向边。")
    print(f"📁 节点文件: {nodes_csv}")
    print(f"📁 边文件:   {edges_csv}")
    print(f"📁 图文件:   {graph_pt}")

    return nodes_csv, edges_csv, graph_pt