# FLASH 激光聚变模拟的 NN 加速方案

## 一、项目背景

### 1.1 FLASH 模拟特点

| 特性 | 描述 |
|------|------|
| **模拟类型** | 2D 激光诱导惯性约束聚变 |
| **物理过程** | 辐射流体力学 + 激光能量沉积 |
| **网格** | PARAMESH 自适应网格细化 (AMR) |
| **变量数** | 9个 (dens, pres, tele, tion, trad, velx, vely, depo, temp) |
| **时间步长** | 自适应，~10⁻¹³ - 10⁻¹⁰ 秒 |
| **总时长** | 10 纳秒 |
| **数据量** | 21个时间步输出 |

### 1.2 加速目标

```
传统 FLASH 模拟:
  每个时间步需要:
  - 求解流体力学方程 (Riemann solver)
  - 激光射线追踪 (ray tracing)
  - 电子热传导 (flux-limited diffusion)
  - 状态方程计算

NN 加速目标:
  用神经网络学习:
  state(t) → state(t + Δt)

  实现速度提升: 10-100倍
```

---

## 二、数据结构分析

### 2.1 FLASH 数据格式

```
HDF5 文件结构:
├── scalar integers: nxb=32, nyb=32, nzb=1, nstep, ...
├── scalar reals: time, dt, ...
├── block size [68, 3]: 每个 block 的物理尺寸
├── bounding box [68, 3, 2]: 每个 block 的边界
├── refine level [68]: AMR 细化层级 (1-3)
├── dens [68, 1, 32, 32]: 密度
├── pres [68, 1, 32, 32]: 压力
├── tele [68, 1, 32, 32]: 电子温度
├── tion [68, 1, 32, 32]: 离子温度
├── trad [68, 1, 32, 32]: 辐射温度
├── velx [68, 1, 32, 32]: X速度
├── vely [68, 1, 32, 32]: Y速度
├── depo [68, 1, 32, 32]: 激光能量沉积
└── temp [68, 1, 32, 32]: 总温度
```

### 2.2 与 MeshGraphNets 数据的对比

| 方面 | MeshGraphNets (流体) | FLASH (激光聚变) |
|------|---------------------|------------------|
| **网格** | 固定三角形网格 | AMR 块结构网格 |
| **节点数** | 固定 | 动态 (68 blocks) |
| **节点特征** | 速度(2) + 类型(9) = 11维 | 密度、温度、压力、速度... |
| **边特征** | 距离 + 相对坐标 = 3维 | 块间连接关系 |
| **物理** | 单温流体 | 多温度等离子体 |
| **边界** | 固定边界 | 反射/流出边界 |

---

## 三、NN 加速方案设计

### 3.1 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    FLASH NN 加速器                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段1: 数据预处理 (Data Preprocessing)                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 读取 HDF5 文件，提取物理变量                      │   │
│  │ 2. 将 AMR 块结构转换为图表示                        │   │
│  │ 3. 归一化处理 (处理数量级差异)                       │   │
│  │ 4. 构造时间序列样本对                               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段2: 模型训练 (Model Training)                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 选项A: 图神经网络 (类似 MeshGraphNets)               │   │
│  │  - 将 AMR 块作为图的节点                             │   │
│  │  - 块间连接作为边                                    │   │
│  │  - 消息传递模拟物理相互作用                         │   │
│  │                                                      │   │
│  │ 选项B: U-Net / ConvNet (转换为规则网格)             │   │
│  │  - 将 AMR 插值到均匀网格                            │   │
│  │  - 使用卷积神经网络                                 │   │
│  │                                                      │   │
│  │ 选项C: Transformer (注意力机制)                      │   │
│  │  - 处理长程依赖 (激光加热区域)                      │   │
│  │  - 多头注意力捕获空间相关性                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段3: 推理加速 (Inference Acceleration)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. NN 预测多个时间步 (自回归)                        │   │
│  │ 2. 每 N 步用 FLASH 校正 (混合模拟)                   │   │
│  │ 3. 输出加速后的模拟结果                             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、详细方案：基于 MeshGraphNets 的改进

### 4.1 图构造方案

#### 方案 A: 块级图 (Block-Level Graph)

```
将每个 AMR 块作为一个图的节点:

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Block 0    │────│  Block 1    │────│  Block 2    │
│  (level 2)  │     │  (level 2)  │     │  (level 2)  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ↓
                  ┌─────────────┐
                  │  Block 15   │
                  │  (level 1)  │  粗网格块
                  └─────────────┘

节点特征 (每个块的平均值或展平的 32×32):
- dens (密度)
- tele, tion, trad (三温度)
- velx, vely (速度)
- depo (能量沉积)
- pres (压力)

边特征:
- 块间距离 (中心到中心)
- 相对位置 (Δx, Δy)
- 细化层级差
```

#### 方案 B: 细胞级图 (Cell-Level Graph)

```
将每个网格单元作为一个节点 (更精细但计算量大):

对于每个 32×32 的块:
┌───┬───┬───┬───┐
│   │   │   │   │
├───┼───┼───┼───┤
│   │   │   │   │  → 1024 个节点/块
├───┼───┼───┼───┤
│   │   │   │   │
├───┼───┼───┼───┤
│   │   │   │   │
└───┴───┴───┴───┘

优点: 更精细的物理表示
缺点: 计算量大 (68 blocks × 1024 cells = 69,632 节点)
```

**推荐**: 先用方案 A (块级图)，如果精度不够再考虑方案 B。

---

### 4.2 模型架构设计

#### 架构对比

```
┌─────────────────────────────────────────────────────────────┐
│  原始 MeshGraphNets                                         │
├─────────────────────────────────────────────────────────────┤
│  输入: 节点(11维), 边(3维)                                   │
│  Encoder → 15× Processor → Decoder                          │
│  输出: 加速度(2维)                                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  改进的 FLASH-Net                                            │
├─────────────────────────────────────────────────────────────┤
│  输入: 节点(9维物理变量), 边(4维+层级), 时间(1维)           │
│  ↓                                                           │
│  Encoder (物理特征嵌入)                                     │
│  ├─ 节点 MLP: 9维 → 128维                                   │
│  └─ 边 MLP: 4维 → 128维                                     │
│  ↓                                                           │
│  Processor (多尺度消息传递)                                 │
│  ├─ 15-30 层 GN 块                                         │
│  ├─ 跨层级连接 (处理 AMR)                                  │
│  └─ 注意力机制 (捕获激光加热)                              │
│  ↓                                                           │
│  Decoder (物理预测)                                         │
│  ├─ 分别预测: dens, tele, tion, velx, vely                 │
│  └─ 或预测增量: Δdens, Δtele, Δtion, Δvelx, Δvely           │
│  ↓                                                           │
│  物理约束层                                                  │
│  ├─ 质量守恒: Σ(dens·V) = const                             │
│  ├─ 能量守恒: 检查预测的能量变化                           │
│  └─ 边界条件: 强制反射/流出边界                            │
└─────────────────────────────────────────────────────────────┘
```

#### 代码框架

```python
class FLASHNet(nn.Module):
    """
    基于图神经网络的 FLASH 加速器
    """
    def __init__(self, node_input_size, edge_input_size, hidden_size=128):
        super().__init__()

        # Encoder
        self.node_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
        self.edge_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)

        # Processor with multi-scale connections
        self.processor_layers = nn.ModuleList([
            MultiScaleGNBlock(hidden_size) for _ in range(15)
        ])

        # Decoder (predict all variables)
        self.decoder = nn.ModuleDict({
            'dens': build_mlp(hidden_size, hidden_size, 1, lay_norm=False),
            'tele': build_mlp(hidden_size, hidden_size, 1, lay_norm=False),
            'tion': build_mlp(hidden_size, hidden_size, 1, lay_norm=False),
            'velx': build_mlp(hidden_size, hidden_size, 1, lay_norm=False),
            'vely': build_mlp(hidden_size, hidden_size, 1, lay_norm=False),
        })

    def forward(self, graph):
        # Encode
        graph.x = self.node_encoder(graph.x)
        graph.edge_attr = self.edge_encoder(graph.edge_attr)

        # Process
        for layer in self.processor_layers:
            graph = layer(graph)

        # Decode (predict deltas)
        outputs = {}
        for var, decoder in self.decoder.items():
            outputs[var] = decoder(graph.x)

        return outputs


class MultiScaleGNBlock(nn.Module):
    """
    支持跨层级连接的 GN 块 (处理 AMR)
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.edge_block = EdgeBlock(...)
        self.node_block = NodeBlock(...)
        self.cross_level_attention = CrossLevelAttention(...)

    def forward(self, graph):
        # 标准消息传递
        graph = self.edge_block(graph)
        graph = self.node_block(graph)

        # 跨层级连接 (连接不同细化层级的块)
        graph = self.cross_level_attention(graph)

        # 残差连接
        ...

        return graph
```

---

### 4.3 输入输出设计

#### 输入特征

| 特征类型 | 变量 | 维度 | 说明 |
|---------|------|------|------|
| **物理状态** | dens | 1 | 密度 |
| | tele, tion, trad | 3 | 三温度 |
| | velx, vely | 2 | 速度 |
| | depo | 1 | 能量沉积率 |
| | pres | 1 | 压力 |
| **位置信息** | x, y (归一化) | 2 | 空间位置 |
| **时间信息** | t (归一化) | 1 | 当前时间 |
| **网格信息** | refine_level | 1 | AMR 层级 |
| **材料类型** | material_type (one-hot) | 2 | DT vs 真空 |
| **总计** | | 14 | 节点特征维度 |

#### 输出预测

| 预测目标 | 变量 | 维度 | 说明 |
|---------|------|------|------|
| **预测增量** | Δdens | 1 | 密度变化 |
| | Δtele, Δtion, Δtrad | 3 | 温度变化 |
| | Δvelx, Δvely | 2 | 速度变化 |
| | Δdepo | 1 | 能量沉积变化 |
| **总计** | | 7 | 输出维度 |

或者：
| **预测下一状态** | dens_next | 1 | 下一时刻密度 |
| | tele_next, tion_next, trad_next | 3 | 下一时刻温度 |
| | velx_next, vely_next | 2 | 下一时刻速度 |
| | depo_next | 1 | 下一时刻能量沉积 |
| **总计** | | 7 | 输出维度 |

---

### 4.4 归一化策略

FLASH 数据的数值范围差异很大，需要仔细归一化：

```python
class FLASHNormalizer:
    """
    物理变量的归一化器
    """
    def __init__(self):
        # 对数归一化 (用于跨越数量级的变量)
        self.log_norm_vars = ['dens', 'depo', 'pres']

        # 标准归一化 (用于范围有限的变量)
        self.std_norm_vars = ['velx', 'vely']

        # 温度特殊处理 (K → 对数 + 偏移)
        self.temp_norm_vars = ['tele', 'tion', 'trad']

    def normalize(self, data, var_name):
        if var_name in self.log_norm_vars:
            return np.log10(data + 1e-10)  # 避免log(0)
        elif var_name in self.temp_norm_vars:
            return np.log10(data + 1.0) - 2.0  # 假设初始温度 ~100K
        else:
            return (data - self.mean) / self.std

    def denormalize(self, norm_data, var_name):
        ...
```

#### 典型数值范围

| 变量 | 最小值 | 最大值 | 归一化方法 |
|------|--------|--------|-----------|
| dens (g/cm³) | 1×10⁻⁶ | 0.25 | log₁₀ |
| tele (K) | 20 | 1×10⁷ | log₁₀ |
| velx (cm/s) | -1×10⁷ | 1×10⁷ | 标准归一化 |
| depo (erg/cm³/s) | 0 | 1×10²⁰ | log₁₀ |

---

## 五、实现步骤

### 步骤 1: 数据加载与预处理

```python
# data_loader.py
import h5py
import numpy as np
import torch
from torch_geometric.data import Data

def load_flash_data(filepath):
    """读取单个 HDF5 文件"""
    f = h5py.File(filepath, 'r')

    # 读取元数据
    time = f['real scalars'][0][1]
    nblocks = f['integer scalars'][3][1]  # globalnumblocks

    # 读取块信息
    block_size = f['block size'][:]  # [nblocks, 3]
    bounding_box = f['bounding box'][:]  # [nblocks, 3, 2]
    refine_level = f['refine level'][:]  # [nblocks]

    # 读取物理变量
    dens = f['dens'][:]  # [nblocks, 1, 32, 32]
    tele = f['tele'][:]
    velx = f['velx'][:]
    vely = f['vely'][:]
    depo = f['depo'][:]

    return {
        'time': time,
        'nblocks': nblocks,
        'block_size': block_size,
        'bounding_box': bounding_box,
        'refine_level': refine_level,
        'dens': dens,
        'tele': tele,
        'velx': velx,
        'vely': vely,
        'depo': depo,
    }


def blocks_to_graph(data):
    """将 AMR 块转换为图结构"""

    # 计算块的平均值 (作为节点特征)
    node_features = []

    for i in range(data['nblocks']):
        # 取块内平均值
        dens_avg = data['dens'][i, 0, :, :].mean()
        tele_avg = data['tele'][i, 0, :, :].mean()
        velx_avg = data['velx'][i, 0, :, :].mean()
        vely_avg = data['vely'][i, 0, :, :].mean()
        depo_avg = data['depo'][i, 0, :, :].mean()

        # 块中心位置 (归一化)
        bbox = data['bounding_box'][i]  # [3, 2]
        x_center = (bbox[0, 0] + bbox[0, 1]) / 2
        y_center = (bbox[1, 0] + bbox[1, 1]) / 2

        # 组装节点特征
        features = [
            dens_avg, tele_avg, velx_avg, vely_avg, depo_avg,
            x_center, y_center,
            data['refine_level'][i],
            data['time']
        ]
        node_features.append(features)

    node_features = torch.tensor(node_features, dtype=torch.float32)

    # 构造边 (相邻块连接)
    edge_index, edge_attr = build_edges(data)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


def build_edges(data):
    """构造块间的边连接"""

    edge_index = []
    edge_attr = []

    # 获取所有块的边界
    bboxes = data['bounding_box']  # [nblocks, 3, 2]
    levels = data['refine_level']  # [nblocks]

    for i in range(data['nblocks']):
        for j in range(i + 1, data['nblocks']):
            # 检查块 i 和 j 是否相邻
            if is_adjacent(bboxes[i], bboxes[j]):
                # 添加边
                edge_index.append([i, j])
                edge_index.append([j, i])

                # 边特征: 距离, 相对位置, 层级差
                center_i = get_center(bboxes[i])
                center_j = get_center(bboxes[j])
                distance = np.linalg.norm(center_i[:2] - center_j[:2])
                rel_pos = center_j[:2] - center_i[:2]
                level_diff = abs(levels[i] - levels[j])

                edge_attr.append([distance, rel_pos[0], rel_pos[1], level_diff])
                edge_attr.append([distance, -rel_pos[0], -rel_pos[1], level_diff])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    return edge_index, edge_attr


def is_adjacent(bbox1, bbox2):
    """检查两个块是否相邻 (共享边界)"""
    # 简化版本: 检查是否在某个维度接触
    for dim in range(2):  # 只检查 x, y
        if (abs(bbox1[dim, 0] - bbox2[dim, 1]) < 1e-10 or
            abs(bbox1[dim, 1] - bbox2[dim, 0]) < 1e-10):
            # 在这个维度接触，检查另一个维度是否重叠
            other_dim = 1 - dim
            if not (bbox1[other_dim, 1] < bbox2[other_dim, 0] or
                    bbox1[other_dim, 0] > bbox2[other_dim, 1]):
                return True
    return False


def get_center(bbox):
    """获取块中心坐标"""
    return np.array([
        (bbox[0, 0] + bbox[0, 1]) / 2,
        (bbox[1, 0] + bbox[1, 1]) / 2,
        (bbox[2, 0] + bbox[2, 1]) / 2,
    ])
```

---

### 步骤 2: 创建数据集类

```python
# dataset.py
import os
from torch.utils.data import Dataset
from torch_geometric.data import Data

class FLASHDataset(Dataset):
    """FLASH 模拟数据集"""

    def __init__(self, data_dir, file_pattern='laser_fusion_2d_long_hdf5_plt_cnt_*'):
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        self.data_cache = []

        # 预加载所有数据
        for filepath in self.files:
            self.data_cache.append(load_flash_data(filepath))

    def __len__(self):
        return len(self.data_cache) - 1  # 返回样本对数量

    def __getitem__(self, idx):
        """
        返回连续两个时间步的图数据
        输入: t 时刻的状态
        目标: t+1 时刻的状态增量
        """
        current_data = self.data_cache[idx]
        next_data = self.data_cache[idx + 1]

        # 构造图
        current_graph = blocks_to_graph(current_data)
        next_graph = blocks_to_graph(next_data)

        # 计算增量 (作为训练目标)
        delta = next_graph.x - current_graph.x

        return current_graph, delta
```

---

### 步骤 3: 模型定义

```python
# model.py
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class FLASHNet(nn.Module):
    """FLASH 加速器神经网络"""

    def __init__(self, node_input_size=9, edge_input_size=4, hidden_size=128):
        super().__init__()

        # Normalizers
        self.node_normalizer = OnlineNormalizer(node_input_size)
        self.edge_normalizer = OnlineNormalizer(edge_input_size)
        self.output_normalizer = OnlineNormalizer(node_input_size)

        # Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Processor
        self.processor_blocks = nn.ModuleList([
            GNBlock(hidden_size) for _ in range(15)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, node_input_size),
        )

    def forward(self, graph):
        # Normalize
        x = self.node_normalizer(graph.x, self.training)
        edge_attr = self.edge_normalizer(graph.edge_attr, self.training)

        # Encode
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Process
        for block in self.processor_blocks:
            x, edge_attr = block(x, edge_attr, graph.edge_index)

        # Decode
        delta_pred = self.decoder(x)

        # Denormalize
        if self.training:
            return self.output_normalizer(delta_pred, normalize=True), \
                   self.output_normalizer(graph.y, normalize=True)
        else:
            delta = self.output_normalizer.inverse(delta_pred)
            return graph.x + delta


class GNBlock(nn.Module):
    """图神经网络块"""

    def __init__(self, hidden_size):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x, edge_attr, edge_index):
        # Edge update
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_attr_new = self.edge_mlp(edge_input)
        edge_attr = edge_attr + edge_attr_new  # 残差

        # Node update
        from torch_scatter import scatter_add
        agg = scatter_add(edge_attr, col, dim=0, out_size=x.size(0))
        node_input = torch.cat([x, agg], dim=1)
        x_new = self.node_mlp(node_input)
        x = x + x_new  # 残差

        return x, edge_attr
```

---

### 步骤 4: 训练脚本

```python
# train.py
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

def train():
    # 加载数据
    dataset = FLASHDataset(
        data_dir='/vepfs-mlp2/c20250518/zihanbian/flash/laser_fusion_2d/test2/object'
    )

    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # 创建模型
    model = FLASHNet(
        node_input_size=9,  # dens, tele, velx, vely, depo, x, y, level, t
        edge_input_size=4,  # distance, rel_x, rel_y, level_diff
        hidden_size=128
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(100):
        model.train()
        train_loss = 0

        for batch in train_loader:
            graph, target = batch
            graph = graph.cuda()
            target = target.cuda()

            optimizer.zero_grad()

            # 前向传播
            pred, target_norm = model(graph)

            # 损失 (对不同变量加权)
            loss = compute_weighted_loss(pred, target_norm)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                graph, target = batch
                graph = graph.cuda()
                target = target.cuda()

                pred, target_norm = model(graph)
                loss = compute_weighted_loss(pred, target_norm)
                val_loss += loss.item()

        print(f'Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.6f}, '
              f'Val Loss = {val_loss/len(val_loader):.6f}')


def compute_weighted_loss(pred, target):
    """计算加权损失 (不同物理变量权重不同)"""
    # 假设前5列是物理变量 (dens, tele, velx, vely, depo)
    weights = torch.tensor([1.0, 2.0, 0.5, 0.5, 1.0]).cuda()  # 温度更重要

    squared_error = (pred - target) ** 2
    weighted_error = squared_error * weights.unsqueeze(0)

    return weighted_error.mean()


if __name__ == '__main__':
    train()
```

---

### 步骤 5: 评估与可视化

```python
# evaluate.py
def evaluate(model, data_dir, num_steps=100):
    """评估模型的长时间预测能力"""

    model.eval()

    # 加载初始状态
    initial_data = load_flash_data(
        os.path.join(data_dir, 'laser_fusion_2d_long_hdf5_plt_cnt_0000')
    )

    # 自回归预测
    current_state = blocks_to_graph(initial_data)
    predictions = [current_state]

    with torch.no_grad():
        for step in range(num_steps):
            # 预测下一状态
            next_state = model(current_state)

            # 可选: 每 N 步用 FLASH 校正
            if step % 10 == 0:
                corrected = flash_correction(next_state, ...)
                next_state = corrected

            predictions.append(next_state)
            current_state = next_state

    return predictions


def visualize_predictions(predictions, ground_truth):
    """可视化预测结果"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    variables = ['dens', 'tele', 'tion', 'velx', 'vely', 'depo']
    names = ['Density (g/cm³)', 'Electron Temp (K)', 'Ion Temp (K)',
             'X Velocity (cm/s)', 'Y Velocity (cm/s)', 'Deposition (erg/cm³/s)']

    for i, (var, name) in enumerate(zip(variables, names)):
        ax = axes[i // 3, i % 3]

        # 绘制预测
        for t, pred in enumerate(predictions[::10]):
            # ... 绘制逻辑 ...

        # 绘制真实值
        # ... 绘制逻辑 ...

        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    plt.savefig('predictions.png')
```

---

## 六、进阶优化

### 6.1 混合模拟策略

```
┌─────────────────────────────────────────────────────────────┐
│              混合 FLASH-NN 模拟策略                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  t=0                                                        │
│   ↓                                                         │
│  ┌─────────┐                                               │
│  │  FLASH  │  初始化 (用传统模拟)                          │
│  └─────────┘                                               │
│     ↓                                                       │
│  ┌─────────────────────────────────────────────┐           │
│  │                                            │           │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │           │
│  │  │ NN │ →│ NN │ →│ NN │ →│ NN │ →│ NN │  │  NN加速段  │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘  │  (快速)   │
│  │                                            │           │
│  └─────────────────────────────────────────────┘           │
│                  ↓ 每 N 步                                  │
│              ┌─────────┐                                   │
│              │  FLASH  │  校正 (用传统模拟检查和修正)       │
│              └─────────┘                                   │
│                  ↓                                         │
│              重复 NN 预测...                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

优势:
1. NN 提供快速预测
2. FLASH 定期校正，防止误差累积
3. 平衡速度和精度
```

### 6.2 物理约束损失

```python
def physics_informed_loss(pred, target, graph):
    """添加物理约束的损失函数"""

    # 基础 MSE 损失
    loss = F.mse_loss(pred, target)

    # 质量守恒约束
    # 总质量应该守恒 (除非有源项)
    mass_pred = pred[:, 0].sum()  # dens
    mass_target = target[:, 0].sum()
    loss += 0.1 * (mass_pred - mass_target) ** 2

    # 能量约束
    # 总能量变化应该等于激光沉积能量
    energy_pred = compute_total_energy(pred)
    energy_input = graph.depo.sum()
    loss += 0.1 * (energy_pred - energy_input) ** 2

    # 边界条件约束
    # 反射边界: 法向速度为0
    # 流出边界: 梯度为0
    loss += boundary_condition_loss(pred, graph)

    return loss
```

### 6.3 不确定性量化

```python
class ProbabilisticFLASHNet(nn.Module):
    """带不确定性估计的模型"""

    def __init__(self, ...):
        super().__init__()
        self.mean_net = FLASHNet(...)
        self.var_net = FLASHNet(...)  # 预测方差

    def forward(self, graph):
        mean = self.mean_net(graph)
        log_var = self.var_net(graph)
        return mean, torch.exp(log_var)

    def loss(self, mean, log_var, target):
        """负对数似然损失"""
        mse = (mean - target) ** 2
        nll = mse / (2 * torch.exp(log_var)) + log_var / 2
        return nll.mean()
```

---

## 七、项目目录结构

```
/vepfs-mlp2/c20250518/zihanbian/nnforpde/flash_accelerator/
├── data/
│   ├── __init__.py
│   ├── loader.py              # HDF5 数据加载
│   ├── preprocessing.py       # 归一化、图构造
│   └── dataset.py             # PyTorch Dataset
│
├── model/
│   ├── __init__.py
│   ├── flash_net.py           # 主模型
│   ├── blocks.py              # GN 块
│   ├── attention.py           # 注意力机制
│   └── normalizer.py          # 归一化器
│
├── utils/
│   ├── __init__.py
│   ├── physics.py             # 物理约束
│   ├── visualization.py       # 可视化
│   └── metrics.py             # 评估指标
│
├── train.py                   # 训练脚本
├── evaluate.py                # 评估脚本
├── hybrid_simulate.py         # 混合模拟
├── config.py                  # 配置文件
└── README.md                  # 说明文档
```

---

## 八、关键挑战与解决方案

| 挑战 | 解决方案 |
|------|---------|
| **AMR 结构复杂** | 将块作为节点，用层级差作为边特征 |
| **数值范围差异大** | 对数归一化 + 分变量归一化 |
| **多温度物理** | 分别预测电子、离子、辐射温度 |
| **激光能量沉积** | 添加 depo 作为输入特征，或单独建模 |
| **长时间预测稳定性** | 混合模拟 (定期 FLASH 校正) |
| **边界条件** | 在损失函数中强制边界约束 |
| **守恒律** | 添加物理约束损失 |

---

## 九、预期成果

1. **加速比**: 10-100倍 (取决于精度要求)
2. **精度**: 相对误差 < 5% (在混合模拟模式下)
3. **应用**:
   - 快速参数扫描 (激光能量、目标形状等)
   - 实时控制与优化
   - 不确定性量化

---

## 十、下一步行动

1. ✅ **数据探索** (已完成)
   - 理解 FLASH 数据结构
   - 分析物理变量范围

2. **Phase 1: 基础实现** (1-2周)
   - 数据加载器
   - 图构造
   - 简单模型训练

3. **Phase 2: 模型优化** (2-3周)
   - 添加物理约束
   - 改进架构
   - 超参数调优

4. **Phase 3: 混合模拟** (1-2周)
   - 实现校正机制
   - 长时间稳定性测试
   - 性能评估

5. **Phase 4: 应用与部署** (持续)
   - 参数扫描
   - 可视化工具
   - 文档与示例

---

**有任何问题随时问我！我们可以一步步实现这个方案。**
