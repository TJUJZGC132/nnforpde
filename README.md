# Neural Networks for PDE

## MeshGraphNets_pytorch 项目解构

### 项目概述

这是一个 **DeepMind MeshGraphNets** 的 PyTorch 实现，用于学习基于网格的物理模拟（流体动力学）。

**核心思想：** 用图神经网络替代传统数值求解器，实现 **10-100倍加速**。

**仓库地址：** `https://github.com/echowve/meshGraphNets_pytorch`
**本地路径：** `/vepfs-mlp2/c20250518/zihanbian/meshgraph/meshGraphNets_pytorch/`

---

## 目录结构

```
meshgraph/meshGraphNets_pytorch/
├── model/                      # 核心模型
│   ├── model.py               # Encoder-Processor-Decoder 架构
│   ├── simulator.py           # 主模拟器封装
│   └── blocks.py              # GNN 基础模块
│
├── dataset/                    # 数据加载
│   └── fpc.py                 # 流体数据集实现
│
├── utils/                      # 工具函数
│   ├── normalization.py       # 在线归一化
│   ├── noise.py               # 噪声注入
│   └── utils.py               # 节点类型定义
│
├── train.py                    # 单 GPU 训练
├── train_ddp.py                # 多 GPU 分布式训练
├── rollout.py                  # 长时序预测评估
├── parse_tfrecord.py           # 数据格式转换 (TFRecord → PyTorch)
└── render_results.py           # 结果可视化 (生成 MP4)
```

---

## 核心架构：MeshGraphNets

### 模型流程图

```
                    ┌─────────────────────────────────────┐
                    │         输入特征                     │
                    │  • 节点: [速度(2) + 类型one-hot(9)] │
                    │  • 边: [距离, 相对坐标x, 相对坐标y]  │
                    └─────────────────────────────────────┘
                                    ↓
                    ┌─────────────────────────────────────┐
                    │         ENCODER                     │
                    │  MLP: 11维 → 128维 (节点)           │
                    │  MLP: 3维 → 128维 (边)              │
                    └─────────────────────────────────────┘
                                    ↓
                    ┌─────────────────────────────────────┐
                    │      PROCESSOR (×15层)              │
                    │  ┌─────────────────────────────┐   │
                    │  │  EdgeBlock: 边特征更新       │   │
                    │  │  e_ij = MLP([x_i, x_j, e_ij])│   │
                    │  └─────────────────────────────┘   │
                    │              ↓                      │
                    │  ┌─────────────────────────────┐   │
                    │  │  NodeBlock: 节点特征更新     │   │
                    │  │  x_i = MLP([x_i, Σe_ji])    │   │
                    │  └─────────────────────────────┘   │
                    │         (残差连接)                  │
                    └─────────────────────────────────────┘
                                    ↓
                    ┌─────────────────────────────────────┐
                    │         DECODER                     │
                    │  MLP: 128维 → 2维 (加速度预测)      │
                    └─────────────────────────────────────┘
                                    ↓
                    ┌─────────────────────────────────────┐
                    │         输出                         │
                    │  预测加速度 → 速度更新               │
                    └─────────────────────────────────────┘
```

---

## 关键代码文件说明

### 模型模块 (`/model/`)

#### [model/model.py](../meshgraph/meshGraphNets_pytorch/model/model.py)
核心架构组件：

| 类 | 功能 |
|---|------|
| `build_mlp()` | 创建 4 层 MLP + LayerNorm |
| `Encoder` | 编码节点/边特征 → 128 维潜在空间 |
| `GnBlock` | 图网络块（EdgeBlock + NodeBlock + 残差连接） |
| `Decoder` | 解码潜在特征 → 2D 加速度 |
| `EncoderProcesserDecoder` | 完整流水线：Encoder → 15×GN → Decoder |

#### [model/simulator.py](../meshgraph/meshGraphNets_pytorch/model/simulator.py)
主模拟器封装类：

- 管理三个归一化器（节点、边、输出）
- Xavier 权重初始化
- 两种前向模式：
  - **训练模式**: 添加噪声，预测归一化加速度
  - **推理模式**: 预测速度更新（去归一化）

#### [model/blocks.py](../meshgraph/meshGraphNets_pytorch/model/blocks.py)
GNN 基础模块：

| 类 | 功能 |
|---|------|
| `EdgeBlock` | 更新边特征: `e_ij = MLP([x_i, x_j, e_ij])` |
| `NodeBlock` | 更新节点特征: `x_i = MLP([x_i, Σe_ji])` |

---

### 数据集模块 (`/dataset/`)

#### [dataset/fpc.py](../meshgraph/meshGraphNets_pytorch/dataset/fpc.py)
流体粒子容器数据集实现：

- 使用**内存映射文件** (.dat) 高效加载速度序列
- 从压缩 .npz 文件加载静态网格数据
- 每个样本返回：
  - `x`: [node_type, velocity] - 节点特征
  - `pos`: 网格位置
  - `face`: 三角形连接（用于 PyG 的 FaceToEdge 变换）
  - `y`: 下一个时间步的速度（目标）

---

### 工具模块 (`/utils/`)

#### [utils/utils.py](../meshgraph/meshGraphNets_pytorch/utils/utils.py)
节点类型枚举：

```python
NodeType:
    NORMAL = 0          # 正常流体节点 (参与训练)
    OBSTACLE = 1        # 障碍物边界
    AIRFOIL = 2         # 翼型表面
    HANDLE = 3          # 控制节点
    INFLOW = 4          # 入口边界
    OUTFLOW = 5         # 出口边界 (参与训练)
    WALL_BOUNDARY = 6   # 墙壁边界
    SIZE = 9            # one-hot 编码总维度
```

#### [utils/normalization.py](../meshgraph/meshGraphNets_pytorch/utils/normalization.py)
在线统计归一化器：

- 维护运行均值和标准差（最多 100 万批次）
- 归一化: `(x - mean) / (std + epsilon)`
- 用于节点特征、边特征和输出加速度

#### [utils/noise.py](../meshgraph/meshGraphNets_pytorch/utils/noise.py)
噪声注入策略：

- **训练时**: 向 NORMAL 节点添加高斯噪声 (std=2e-2)
- **推理时**: 不添加噪声

---

### 训练脚本

#### [train.py](../meshgraph/meshGraphNets_pytorch/train.py)
单 GPU 训练脚本：

- Adam 优化器 (lr=1e-4)
- TensorBoard 日志记录
- 模型检查点保存（最佳验证模型）
- 配置: 100 epochs, batch_size=20, noise_std=2e-2
- MSE 损失（仅 NORMAL 和 OUTFLOW 节点）

#### [train_ddp.py](../meshgraph/meshGraphNets_pytorch/train_ddp.py)
多 GPU 分布式训练：

- PyTorch DistributedDataParallel 封装
- DistributedSampler 数据分片
- NCCL 后端 GPU 通信
- 命令行参数配置

#### [rollout.py](../meshgraph/meshGraphNets_pytorch/rollout.py)
长时序自回归预测评估：

- 生成多步预测（无真实值）
- 计算轨迹累积 RMSE 误差
- 保存预测结果为 pickle 文件
- 边界条件使用真实值（非 NORMAL 节点）

---

### 数据处理

#### [parse_tfrecord.py](../meshgraph/meshGraphNets_pytorch/parse_tfrecord.py)
TFRecord → PyTorch 格式转换：

- 读取 DeepMind 的 TFRecord 文件
- 输出内存映射的 .dat 文件（速度）
- 创建压缩的 .npz 文件（网格元数据）
- 处理轨迹索引

#### [render_results.py](../meshgraph/meshGraphNets_pytorch/render_results.py)
从 rollout 结果生成可视化视频：

- 并行渲染（ProcessPoolExecutor）
- 并排比较（目标 vs 预测）
- Matplotlib 三角剖分网格可视化
- 输出 MP4 视频

---

## 训练管线

```
┌──────────────────────────────────────────────────────────────┐
│ 1. 数据准备                                                   │
│    TFRecord → parse_tfrecord.py → .dat + .npz                │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. 数据加载                                                   │
│    FpcDataset → FaceToEdge → Cartesian → Distance            │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. 训练循环 (train.py)                                        │
│    • 加噪声 (std=0.02)                                        │
│    • 前向传播 → 预测加速度                                    │
│    • 计算损失 (仅 NORMAL + OUTFLOW 节点)                      │
│    • 反向传播 → Adam 优化 (lr=1e-4)                           │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. 评估 (rollout.py)                                         │
│    • 自回归多步预测                                          │
│    • 计算 RMSE 误差                                          │
│    • 保存预测结果                                            │
└──────────────────────────────────────────────────────────────┘
```

---

## 数据流向

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 数据加载                                                   │
│    FpcDataset[index] → Data(x, pos, face, y)                │
│    - x: [node_type, v_x, v_y]                               │
│    - y: [next_v_x, next_v_y]                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 图变换                                                     │
│    T.Compose([FaceToEdge, Cartesian, Distance])             │
│    - 将面转换为边                                            │
│    - 添加边属性: [rel_x, rel_y, distance]                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 批处理                                                     │
│    DataLoader → 带有 edge_index 连接的批处理数据             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 训练前向传播                                               │
│    a) 向速度添加噪声（仅训练）                                │
│    b) one-hot 编码 node_type → 9D                           │
│    c) 拼接: [velocity(2), one_hot(9)] = 11D                 │
│    d) 用运行统计归一化节点特征                                │
│    e) 归一化边属性                                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 模型前向                                                   │
│    Encoder (MLP: 11/3 → 128)                                │
│    ↓                                                         │
│    15× GN Blocks (EdgeBlock → NodeBlock + 残差)             │
│    ↓                                                         │
│    Decoder (MLP: 128 → 2)                                   │
│    输出: predicted_acceleration_normalized                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. 损失计算                                                   │
│    - 计算 target_acceleration = next_v - current_v          │
│    - 归一化 target_acceleration                             │
│    - 掩码: 仅 NORMAL 和 OUTFLOW 节点                         │
│    - MSE Loss: mean((pred - target)²)                       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. 反向传播                                                   │
│    Loss.backward() → Adam.step()                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. 推理 (rollout.py)                                         │
│    - 不添加噪声                                               │
│    - 预测加速度                                               │
│    - 去归一化: acc = denormalize(pred_acc_norm)              │
│    - 积分: v_next = v_current + acc                          │
│    - 掩码: 用真实值替换边界节点                               │
│    - 自回归重复多步预测                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 节点类型

| 类型 | 值 | 说明 | 是否训练 |
|------|-----|------|---------|
| NORMAL | 0 | 正常流体节点 | ✅ |
| OBSTACLE | 1 | 障碍物边界 | ❌ |
| AIRFOIL | 2 | 翼型表面 | ❌ |
| HANDLE | 3 | 控制节点 | ❌ |
| INFLOW | 4 | 入口边界 | ❌ |
| OUTFLOW | 5 | 出口边界 | ✅ |
| WALL_BOUNDARY | 6 | 墙壁边界 | ❌ |

---

## 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `message_passing_num` | 15 | GNN 层数 |
| `node_input_size` | 11 | 节点特征维度 (2+9) |
| `edge_input_size` | 3 | 边特征维度 |
| `hidden_size` | 128 | 隐藏层维度 |
| `lr` | 1e-4 | 学习率 |
| `batch_size` | 20 | 批大小 |
| `noise_std` | 2e-2 | 训练噪声标准差 |
| `num_epochs` | 100 | 训练轮数 |

---

## 关键设计决策

| 特性 | 说明 |
|------|------|
| **内存效率** | 使用内存映射文件处理大规模数据集 |
| **模块化架构** | 模型/数据/工具清晰分离 |
| **灵活训练** | 支持单 GPU 和多 GPU |
| **在线归一化** | 训练期间自适应数据分布 |
| **图表示** | 使用 PyTorch Geometric 高效消息传递 |
| **噪声注入** | 提高泛化能力，防止过拟合 |
| **残差连接** | GN 块中更好的梯度流动 |
| **掩码损失** | 仅训练自由流动节点（边界条件强制） |

---

## 依赖项

```
matplotlib
numpy
opencv_python
packaging
Pillow
torch
torch_geometric
torch_scatter
tqdm
tensorboard
```

---

## 论文参考

- **DeepMind MeshGraphNets**: [Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks" (ICLR 2021)](https://arxiv.org/abs/2010.03409)

---

## 总结

这是一个设计精良、结构清晰的 MeshGraphNets 实现，适合用于研究流体动力学和其他网格物理模拟任务。代码紧密遵循原始 DeepMind 论文，同时适配了 PyTorch 和 PyTorch Geometric。
