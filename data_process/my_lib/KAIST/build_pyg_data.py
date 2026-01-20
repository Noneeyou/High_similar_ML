import os
import scipy.io as sio
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

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
    label_col: int = None,
    block_size: int = 5000
):
    """
    基于余弦相似度 + KNN 建图（分块计算版，低内存）。
    🚫 自动忽略首列（序号）与标签列。
    ✅ 支持 data.y。
    ✅ 输出 nodes.csv / edges.csv / graph.pt。

    参数:
        csv_path (str): 输入 CSV 文件路径。
        save_dir (str): 图文件输出文件夹。
        num_edges (int): 每个节点连接的邻点数 (KNN数量)。
        label_col (int): 标签列索引（默认 None → 最后一列）。
        block_size (int): 分块计算大小（越小越省内存）。
    返回:
        (nodes_csv, edges_csv, graph_pt)
    """

    # ===================== 1️⃣ 读取数据 =====================
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 找不到输入文件: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"📊 已读取数据: {df.shape}")

    # === 提取标签列 ===
    if label_col is None:
        label_col = df.shape[1] - 1
    y = torch.tensor(df.iloc[:, label_col].values, dtype=torch.long)

    # === 忽略首列与标签列 ===
    drop_cols = [df.columns[0], df.columns[label_col]] if label_col != 0 else [df.columns[0]]
    df_features = df.drop(columns=drop_cols, errors="ignore")
    df_features = df_features.select_dtypes(include=["float", "int"])
    features = df_features.values.astype(np.float32)
    num_nodes = features.shape[0]
    print(f"🧩 使用特征列数: {features.shape[1]} | 节点数: {num_nodes}")

    # ===================== 2️⃣ 分块计算余弦相似度 =====================
    edges = []
    print(f"⚙️ 分块计算余弦相似度，每块 {block_size} 节点，目标邻点数={num_edges}")

    for start in range(0, num_nodes, block_size):
        end = min(num_nodes, start + block_size)
        block = features[start:end]

        # 计算 block 与全体样本的相似度
        sim_part = cosine_similarity(block, features)
        np.fill_diagonal(sim_part[:, start:end], -np.inf)

        # 取前 num_edges 个最相似节点
        for bi in range(sim_part.shape[0]):
            global_i = start + bi
            topk_idx = np.argpartition(sim_part[bi], -num_edges)[-num_edges:]
            for j in topk_idx:
                edges.append((global_i, j))
                edges.append((j, global_i))

        print(f"  ✅ 已处理 {end}/{num_nodes} 节点块")

    edges = np.array(edges, dtype=np.int64)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)

    # ===================== 3️⃣ 构造 PyG Data =====================
    data = Data(x=x, edge_index=edge_index, y=y)

    # ===================== 4️⃣ 保存文件 =====================
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    nodes_csv = os.path.join(save_dir, f"{base_name}_nodes.csv")
    edges_csv = os.path.join(save_dir, f"{base_name}_edges.csv")
    graph_pt = os.path.join(save_dir, f"{base_name}_graph.pt")

    pd.DataFrame(features).to_csv(nodes_csv, index=False)
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(edges_csv, index=False)
    torch.save(data, graph_pt)

    print(f"\n✅ 图构建完成，共 {num_nodes} 个节点，{len(edges)//2} 条无向边。")
    print(f"📁 节点文件: {nodes_csv}")
    print(f"📁 边文件:   {edges_csv}")
    print(f"📁 图文件:   {graph_pt}")

    return nodes_csv, edges_csv, graph_pt

def add_random_masks_to_pyg(
    graph_path: str,
    save_path: str = None,
    ratios: dict = {"train": 0.6, "val": 0.2, "test": 0.2},
    seed: int = 42
):
    """
    向 PyG 格式的图文件中添加随机的 train/val/test 掩码数组。
    
    参数:
        graph_path (str): 原始图文件路径 (.pt)。
        save_path (str): 新文件保存路径；若为 None，则覆盖原文件。
        ratios (dict): 各掩码占比，如 {"train":0.6, "val":0.2, "test":0.2}。
        seed (int): 随机种子，保证可复现。
    
    返回:
        save_path (str): 保存的新图文件路径。
    """

    # ===================== 1️⃣ 读取图文件 =====================
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"❌ 找不到图文件: {graph_path}")
    
    data = torch.load(graph_path)
    num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.shape[0]
    print(f"📦 已加载图，共 {num_nodes} 个节点。")

    # ===================== 2️⃣ 检查已有掩码 =====================
    masks_exist = all(hasattr(data, k + "_mask") for k in ["train", "val", "test"])
    if masks_exist:
        print("⚠️ 掩码已存在，未做修改。")
        return graph_path

    # ===================== 3️⃣ 生成随机掩码 =====================
    np.random.seed(seed)
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    n_train = int(num_nodes * ratios.get("train", 0.6))
    n_val   = int(num_nodes * ratios.get("val", 0.2))
    n_test  = num_nodes - n_train - n_val

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"✅ 掩码生成完成：train={n_train}, val={n_val}, test={n_test}")

    # ===================== 4️⃣ 保存文件 =====================
    if save_path is None:
        save_path = graph_path  # 覆盖原文件
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(data, save_path)
    print(f"💾 新图文件已保存：{save_path}")

    return save_path

def add_random_masks_with_label_split(
    graph_path: str,
    save_path: str = None,
    ratios: dict = {"train": 0.6, "val": 0.2, "test": 0.2},
    train_label_ratio: float = 0.5,
    seed: int = 42
):
    """
    向 PyG 图文件中添加 train/val/test 掩码，
    并在 train 内部划分 train_withlabel_mask / train_nolabel_mask。
    若未指定 save_path，则覆盖原文件。

    参数:
        graph_path (str): 输入图文件路径 (.pt)
        save_path (str): 保存路径；若为 None，则覆盖原文件
        ratios (dict): 掩码占比，如 {"train":0.6, "val":0.2, "test":0.2}
        train_label_ratio (float): 在 train 样本中有标签比例
        seed (int): 随机种子
    返回:
        save_path (str): 保存的新图文件路径
    """

    # ===================== 1️⃣ 加载 PyG 图 =====================
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"❌ 找不到图文件: {graph_path}")

    data = torch.load(graph_path)
    num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.shape[0]
    print(f"📦 已加载图文件，共 {num_nodes} 个节点。")

    # ===================== 2️⃣ 检查是否已有掩码 =====================
    masks_exist = any(hasattr(data, m) for m in ["train_mask", "val_mask", "test_mask"])
    if masks_exist:
        print("⚠️ 已检测到图中存在部分掩码，将不会覆盖已有掩码。")

    # ===================== 3️⃣ 随机划分索引 =====================
    np.random.seed(seed)
    indices = np.random.permutation(num_nodes)

    n_train = int(num_nodes * ratios.get("train", 0.6))
    n_val   = int(num_nodes * ratios.get("val", 0.2))
    n_test  = num_nodes - n_train - n_val

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    # ===================== 4️⃣ 生成主掩码 =====================
    if not hasattr(data, "train_mask"):
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
    if not hasattr(data, "val_mask"):
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask[val_idx] = True
    if not hasattr(data, "test_mask"):
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[test_idx] = True

    print(f"✅ 已生成主掩码：train={n_train}, val={n_val}, test={n_test}")

    # ===================== 5️⃣ 生成训练子掩码 =====================
    np.random.seed(seed + 1)
    n_withlabel = int(n_train * train_label_ratio)
    shuffled_train = np.random.permutation(train_idx)

    withlabel_idx = shuffled_train[:n_withlabel]
    nolabel_idx   = shuffled_train[n_withlabel:]

    data.train_withlabel_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_nolabel_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_withlabel_mask[withlabel_idx] = True
    data.train_nolabel_mask[nolabel_idx] = True

    print(f"🎯 训练集内部划分：with_label={n_withlabel}, no_label={len(nolabel_idx)}")

    # ===================== 6️⃣ 保存逻辑 =====================
    if save_path and save_path.strip():
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(data, save_path)
        print(f"💾 已将含掩码的新图文件保存至：{save_path}")
    else:
        torch.save(data, graph_path)
        save_path = graph_path
        print(f"💾 未指定保存路径，已覆盖原文件：{graph_path}")

    return save_path