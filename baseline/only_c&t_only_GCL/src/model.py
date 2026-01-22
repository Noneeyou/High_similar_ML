import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv

# ========= 1️⃣ 图编码器部分 =========
class GCNEncoder(nn.Module):
    """
    基础图卷积编码器，用于提取结构化特征。
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# ========= 2️⃣ 投影头部分 =========
class MLPHead(nn.Module):
    """
    高质量投影头（SimCLR 标准）
    结构：Linear → LayerNorm → GELU → Linear
    """
    def __init__(self, in_dim, proj_dim):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, proj_dim)
        self.ln1 = nn.LayerNorm(proj_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)  # 非线性提升投影空间表达能力
        x = self.fc2(x)
        return x



# ========= 3️⃣ 封装类：GraphContrastiveLearner =========
class GraphContrastiveLearner(nn.Module):
    """
    图对比学习模块：
    - 内含编码器 + 投影头
    - 提供特征提取与对比损失计算
    """
    def __init__(self, in_dim, hidden_dim, out_dim, proj_dim, tau=0.5):
        super(GraphContrastiveLearner, self).__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, out_dim)
        self.projector = MLPHead(out_dim, proj_dim)
        self.tau = nn.Parameter(torch.tensor(tau))
        self.lambda_bt = 0.2     # Barlow Twins 权重,0.2
        self.lambda_im = 0.5       # InfoMin 权重,0.9
        self.lamba_info_nce = 1

    def forward(self, x, edge_index):
        """
        前向计算，返回编码器特征 h 和投影特征 z
        """
        h = self.encoder(x, edge_index)
        z = self.projector(h)
        return h, z

    def info_nce_loss(self, z1, z2, symmetric: bool = True):
        """
        最终优化版 InfoNCE：
        - 可学习温度 tau
        - 数值稳定 (log_softmax)
        - 对称 InfoNCE（效果更好）
        - Barlow Twins 正则（防 collapse）
        - InfoMin 正则（让表示更分散）
        """
        # --------------- 1) 归一化（余弦相似度） ---------------
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        N = z1.size(0)
        tau = torch.clamp(self.tau, min=0.01, max=2.0)   # 限制范围，防剧烈震荡

        # --------------- 2) 主 InfoNCE ---------------
        logits = torch.matmul(z1, z2.T) / tau
        log_prob_12 = F.log_softmax(logits, dim=1)
        loss_12 = -log_prob_12.diag().mean()

        if symmetric:
            log_prob_21 = F.log_softmax(logits.T, dim=1)
            loss_21 = -log_prob_21.diag().mean()
            info_nce = 0.5 * (loss_12 + loss_21)
        else:
            info_nce = loss_12

        # --------------- 3) Barlow Twins 正则（防 collapse） ---------------
        z1_bn = z1 - z1.mean(dim=0, keepdim=True)
        z2_bn = z2 - z2.mean(dim=0, keepdim=True)

        c = (z1_bn.T @ z2_bn) / N    # 协方差矩阵（proj_dim × proj_dim）

        # 理想矩阵是单位阵 I
        I = torch.eye(c.size(0), device=c.device)
        barlow_loss = ((c - I)**2).sum()

        # --------------- 4) InfoMin 正则（防止表示过于相似） ---------------
        # 鼓励 z1 和 z2 的多样性更高（相反于 InfoMax）
        info_min = (z1 @ z2.T).pow(2).mean()

        # --------------- 5) 融合最终损失 ---------------
        loss = self.lamba_info_nce*info_nce + self.lambda_bt * barlow_loss + self.lambda_im * info_min
        
        return loss

    def compute_loss(self, x1, edge_index1, x2, edge_index2):
        """
        一步式计算（编码 + 投影 + 损失）
        """
        _, z1 = self.forward(x1, edge_index1)
        _, z2 = self.forward(x2, edge_index2)
        loss = self.info_nce_loss(z1, z2)
        return loss

    
def summarize_graph(data):
    # ======== 打印图的基本信息 ========
    print("\n" + "="*60)
    print("🧩 Graph Data Summary")
    print("="*60)

    # 节点信息
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)
    num_features = data.num_features if hasattr(data, 'num_features') else data.x.size(1)
    print(f"📊 节点数量 (num_nodes): {num_nodes}")
    print(f"📈 节点特征维度 (num_features): {num_features}")

    # 边信息
    if hasattr(data, "edge_index"):
        num_edges = data.edge_index.size(1)
        print(f"🔗 边数量 (num_edges): {num_edges}")

        # 检查是否有自环或重复边
        src, dst = data.edge_index
        self_loops = (src == dst).sum().item()
        print(f"🔁 自环数量 (self-loops): {self_loops}")

    # 其他信息
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        print(f"⚙️ 边特征维度 (edge_attr_dim): {data.edge_attr.size(1)}")

    if hasattr(data, "y") and data.y is not None:
        print(f"🎯 标签维度 (y_dim): {data.y.shape}")

    # 打印存储键值
    print(f"\n🧾 Data对象包含字段: {list(data.keys())}")
    print("="*60 + "\n")

def augment_graph(data, feature_drop_prob=0.1, edge_drop_prob=0.1, noise_std=0.01):
    """
    对输入的 PyG Data 对象进行增强
    --------------------------------
    feature_drop_prob : float
        随机mask节点特征的比例
    edge_drop_prob : float
        随机删除边的比例
    noise_std : float
        特征加性噪声标准差
    """
    import torch
    import copy
    import numpy as np

    # 深拷贝一份，防止修改原图
    aug_data = copy.deepcopy(data)

    # ---------- (1) 特征增强 ----------
    x = aug_data.x.clone()

    # 随机mask部分节点特征
    mask = torch.rand_like(x) > feature_drop_prob
    x = x * mask

    # 加噪声（模拟测量误差）
    noise = noise_std * torch.randn_like(x)
    x = x + noise

    aug_data.x = x

    # ---------- (2) 结构增强 ----------
    edge_index = aug_data.edge_index.clone()
    num_edges = edge_index.shape[1]

    # 随机删除一部分边
    keep_mask = torch.rand(num_edges) > edge_drop_prob
    edge_index = edge_index[:, keep_mask]
    aug_data.edge_index = edge_index

    return aug_data

class DownstreamClassifier(nn.Module):
    """
    下游分类头（仅在两层之间加入激活函数）
    """
    def __init__(self, in_dim, num_classes, hidden_dim=128):
        super(DownstreamClassifier, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()            # ⭐ 新增：激活函数
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, h):
        """
        h: encoder 输出的节点特征 [N, in_dim]
        """
        x = self.act(self.fc1(h))       # ⭐ 这里加激活
        logits = self.fc2(x)
        return logits

    def compute_loss(self, h, y, mask):
        """
        下游分类损失（仅对有标签节点）
        """
        logits = self.forward(h)
        idx = mask.nonzero(as_tuple=False).view(-1)
        logits = logits[idx]
        y_true = y[idx]

        return F.cross_entropy(logits, y_true)
