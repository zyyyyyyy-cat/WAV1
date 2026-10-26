# A-V1：面向 WHU-MARS 的 UAD 完整训练改进方案

> **文档用途：** A-V1 的正式研究方案说明，用于统一研究目标、方法组成、数学定义、训练流程、参数配置、实现边界和实验验收标准。  
> **适用范围：** WHU-MARS-1000 首轮完整训练；通过验证后按同一方法扩展至 WHU-MARS-1000-GD 与 WHU-MARS-2337。  
> **训练原则：** 从 ImageNet 预训练 ViT-B/16 初始化，完整执行 UAD 原始训练流程；不以已收敛的 UAD checkpoint 作为训练起点。  
> **方法边界：** 保留 UAD 单分支骨干、原始 CE/WRT/ProCA/GPD 与推理协议；新增模态身份原型判别、Pair/Mixed 图像级对比及软困难身份采样。

---

## 1. 研究目标与基本原则

### 1.1 研究目标

A-V1 希望在不改变 UAD 推理结构的前提下，改善 WHU-MARS 中由 RGB、NIR、TIR 光谱差异以及 Ground/Aerial 视角差异共同造成的身份混淆。现阶段优先关注模态差距，尤其是 TIR 参与的跨模态检索，同时保留原 UAD 对全局身份可分性和跨场景一致性的有效约束。

目标不是让某一项训练损失更低，而是在统一 AS-ReID 评测协议下，提升真实 Query–Gallery 排序指标：

- 总体 mAP、Rank-1、Rank-5、Rank-10、mINP；
- 九种有向模态检索组合（RGB/NIR/TIR Query × RGB/NIR/TIR Gallery）；
- 在数据允许的评测分组中，继续检查 Ground→Aerial 与 Aerial→Ground 的表现。

研究目标是争取所有目标指标稳定提升；方案本身不预先保证增益。必须以完整训练结果判断是否有效。

### 1.2 设计原则

1. **完整训练。** 从 ImageNet 预训练 ViT-B/16 开始，执行原 UAD 的完整训练轮数、优化器、学习率调度、数据增强和采样框架。
2. **保留 UAD 已有机制。** CE、WRT、ProCA、GPD 的定义、特征输入、归一化顺序和原有权重不因 A-V1 而改写。
3. **新增监督直接作用于最终检索描述子。** MP 与 Pair/Mixed 使用同一共享 ViT 输出的 pre-BN CLS 经 L2 归一化后的特征，不引入额外可学习特征分支。
4. **原型记忆不是可学习网络参数。** 原型通过训练特征初始化，并在成功的优化步后使用无梯度 EMA 更新。
5. **采样增强必须保持样本账本不变。** A1 先物化原 PKM 在该 epoch 实际会消耗的 PID-group 调度，再在这些已选 group 之间进行受约束交换以增加困难身份共现；同时在不增删任何样本出现次数的前提下，优先把同一 PID、同一模态的现有样本 occurrence 分组到相机更丰富的 K-group。PID 每 epoch 出现次数、样本 occurrence 多重集、batch 数、每 batch 的 PID 数、每 PID 的模态 K-group 数与 DDP 步数均必须保持不变。
6. **公平对照。** B0 与 A1 从相同 ImageNet 预训练权重开始，采用相同完整训练预算、原始 UAD 配置和评测协议。
7. **首轮冻结方法。** 不再添加专家、额外 BN/LN、可学习 Query、局部 Token、可靠性预测网络或 MINER 的完整方差控制器。

---

## 2. A-V1 的整体结构

A-V1 由原 UAD 基础监督与三项新增机制组成。

| 组成 | 主要职责 | 来源与状态 |
|---|---|---|
| 共享 ViT-B/16、Final LayerNorm、BNNeck、Classifier | 产生统一的身份描述子和 CE logits | UAD 原实现，保留 |
| CE | 通过身份分类器学习训练身份判别 | UAD 原实现，保留 |
| WRT | 对真实 batch 中正负样本距离施加排序约束 | UAD 原实现，保留 |
| ProCA | 先形成模态内身份中心，再将不同模态中心对齐至统一身份中心 | UAD 原实现，保留 |
| GPD | 通过全训练身份的全局原型强化身份间分离 | UAD 原实现，保留 |
| MP（Modality Prototype Discrimination） | 让 Query 在指定目标模态的身份原型集合中进行身份竞争 | A-V1 新增 |
| Pair/Mixed 图像级对比 | 在目标模态 Gallery 和三模态混合 Gallery 中直接约束真实图像匹配关系 | A-V1 新增；借鉴 MINER 多正样本对比思想 |
| 软困难 PID 采样 | 增加相似异身份在同一 batch 中共同出现的概率 | A-V1 新增；借鉴 MINER 软困难身份采样思想 |

总体损失：

$$
\mathcal L_{A} =
\mathcal L_{CE}+\mathcal L_{WRT}
+0.01\mathcal L_{ProCA}+\mathcal L_{GPD}
+\lambda_P(t)\mathcal L_{MP}
+\lambda_C(t)\mathcal L_{img}.
$$

A-V1 不增加推理时特征维度。训练结束后仍使用 768 维 pre-BN CLS，L2 归一化后按原 UAD 评测流程计算欧氏距离。

---

## 3. 原 UAD 的数据流与特征定义

### 3.1 输入与标签对齐

每个训练 iteration 按 RGB、NIR、TIR 组织图像：

$$
X_R,X_N,X_T\in\mathbb R^{B\times3\times H\times W}.
$$

实际每种模态的 batch 结构、每个 rank 的样本数以及全局 batch 解释必须沿用原 UAD PKM sampler 与训练入口，不能仅根据配置名称自行重新解释。WHU/UAD 常用每身份每模态 K=4、每模态 batch 中包含多个身份的 PKM 组织。

模型输入按固定顺序拼接：

$$
X=[X_R;X_N;X_T].
$$

所有标签、相机 ID、模态 ID 与图像顺序逐一对应。模态 ID 固定为 RGB=0、NIR=1、TIR=2。原始数据集 PID 先映射为连续训练类别标签，分类器和 Bank 行索引共用该映射。

### 3.2 模型输出

原 UAD 模型返回：

- `cls_score`：BNNeck 后特征经身份分类器得到的 logits；
- `global_feat`：ViT 最终 LayerNorm 后、BNNeck 前的 CLS；
- `feat`：BNNeck 后特征。

记单图像 pre-BN CLS 为 $f_i\in\mathbb R^{768}$，L2 特征为：

$$
u_i=\frac{f_i}{\max(\|f_i\|_2,\epsilon)}.
$$

特征用途严格区分：

| 路径 | 使用特征 | 说明 |
|---|---|---|
| CE | `cls_score` | 经过 BNNeck 和分类器 |
| WRT | 原 UAD 的 `global_feat` 距离路径 | 保留原实现，不替换为 L2 特征 |
| ProCA | 原实现中的图像特征与模态/身份中心 | 保留原始中心构造顺序 |
| GPD | 原实现中送入 GPD 的 `global_feat`，由 GPD 内部执行归一化 | 保留原温度、Bank 与 EMA 逻辑 |
| MP | $u_i$ | 目标模态原型分类 |
| Pair/Mixed | $u_i$ | 余弦相似度式图像级对比 |
| 推理 | pre-BN CLS，评测时按原配置 L2 | 不使用新增 Bank 或模态标签 |

### 3.3 ProCA 中心计算的归一化顺序

对身份 $y$、模态 $m$ 的 K 张图像，先对每张图像特征 L2，再求均值，最后对均值再次 L2：

$$
\tilde f_{y,m,k}=\operatorname{L2}(f_{y,m,k}),
\quad
a_{y,m}=\frac1K\sum_{k=1}^{K}\tilde f_{y,m,k},
\quad
c_{y,m}=\operatorname{L2}(a_{y,m}).
$$

统一身份中心按原 UAD 代码构造：

$$
g_y=\operatorname{sg}\left[
\operatorname{L2}\left(\frac{c_{y,R}+c_{y,N}+c_{y,T}}3\right)
\right],
$$

其中 `sg` 表示 stop-gradient。ProCA 将模态中心向统一身份中心对齐。A-V1 不修改其中心归一化、detach、损失平均方式或权重 0.01。

### 3.4 原 UAD GPD

GPD 为每个训练身份维护一个全局原型：

$$
P^G\in\mathbb R^{C\times768}.
$$

原始 GPD 对特征与全部身份原型进行温度缩放的分类式判别，温度为 0.03；原型通过 EMA 更新。官方配置的 `GPD_MOMENTUM=0.2` 是新中心更新系数，即旧原型权重 0.8、新中心权重 0.2。

A-V1 保留 GPD 原有 Bank 初始化、有效性处理、损失输入、中心聚合与 EMA 公式，不用新模态 Bank 替换它；B0/A1 共同增加 AMP 成功步门控，只有优化器确实更新参数时才执行原 GPD Bank 更新。

---

## 4. 新增模块一：模态身份原型判别 MP

### 4.1 动机与作用边界

GPD 的全局身份原型有助于统一身份可分性，但它不显式区分目标模态身份参考。MP 为每个训练身份维护 RGB、NIR、TIR 三个模态原型，让 Query 在指定 Gallery 模态的身份集合中竞争。

MP 的目标是补充目标模态条件下的身份竞争，不替代 GPD，也不保证单独改善真实图像 Rank-1。真实图像排序由 Pair/Mixed 与原 WRT 共同补充。

### 4.2 模态原型记忆

$$
P^M\in\mathbb R^{3\times C\times768}.
$$

$P^M[m,y]$ 表示训练身份 $y$ 在模态 $m$ 的历史身份中心，经过 L2 归一化。它不是可学习参数，不参与反向传播。

对当前 batch 中身份 $y$、模态 $m$ 的特征：

$$
c_{y,m}=\operatorname{L2}\left(
\frac1K\sum_{k=1}^{K}\operatorname{L2}(f_{y,m,k})
\right).
$$

若一个 batch 中同一身份在多个 rank 出现，先跨 rank 聚合相同 `(PID, modality)` 的中心，再对每个 Bank 行执行一次 EMA：

$$
P^M_{y,m}\leftarrow
\operatorname{L2}\left(0.8P^M_{y,m}+0.2c_{y,m}\right).
$$

当该行第一次获得有效中心时直接用归一化中心初始化，不把零向量当作有效原型。无样本的身份—模态组合保持无效，不参与对应 softmax 分母。

### 4.3 初始化与 DDP 安全的离线特征提取

训练开始时，模型从 ImageNet 预训练 ViT-B/16 初始化。A1 在第一个训练 epoch 前执行一次无梯度的 MP 原型 bootstrap pass：

1. **数据范围：** 只遍历完整训练划分，不访问测试 Query/Gallery 图像或标签。为此单独构建确定性数据视图，采用与评测一致的 resize/normalize 等确定性预处理，不使用随机翻转、随机擦除等训练增强；使用顺序遍历、`drop_last=False`，不能使用只覆盖本 rank 子集的 `DistributedSampler`。
2. **模型调用：** DDP 训练前由 rank 0 使用未包装的基础模型执行该 pass；若代码结构要求先包装 DDP，则 rank 0 只能调用 `model.module`，不能在其他 rank 不参与 forward 时调用 DDP 包装对象 `model(...)`。其他 rank 等待同步，不参与模型前向。
3. **状态保护：** `eval()` + `torch.inference_mode()`；保存并恢复模型原先的 train/eval 状态。使用独立 DataLoader generator，必要时保存并恢复 Python/NumPy/Torch CPU/CUDA RNG 状态，不能让离线遍历改变正式训练的随机序列。
4. **特征与聚合：** 提取每张训练图像的 pre-BN CLS；每张图像先 L2，再按 `(连续 PID, modality)` 求均值并对均值再次 L2。该聚合仅用于 MP Bank，不改变原 GPD Bank 的初始化。
5. **广播与落盘：** rank 0 生成 `MP_Bank`、`MP_valid`，保存包含数据集划分/manifest 哈希、PID 映射、预处理配置、ImageNet 权重哈希与原型张量哈希的初始化文件；然后将 Bank 与有效性掩码广播给其他 rank，或在共享文件系统上让所有 rank 读取同一文件并核验哈希。所有 rank 在完成后 barrier，再开始训练。若训练中断后从 checkpoint 恢复，则直接恢复保存的 MP Bank/valid，不得重新运行 bootstrap 覆盖训练状态。
6. **不更新模型：** 该 pass 不执行 backward、不调用 optimizer、不推进 scheduler，也不更新 backbone、BNNeck 或 classifier。它只初始化新增记忆，不改变完整训练的起点。

若没有可用的某身份—模态组合，则对应行保持 invalid，不用零向量伪装成有效原型。初始原型来自 ImageNet 特征，身份判别质量可能有限，因此 MP 权重按第 5 个 epoch 达到完整权重的既定 warm-up 逐步启用。

### 4.4 六个跨模态方向

对 $m\ne n$，方向 $m\to n$ 表示：以模态 $m$ 的真实图像特征作为 Query，在目标模态 $n$ 的所有有效身份原型中进行分类。

$$
\mathcal L^P_{m\to n}
=-\frac1{|Q_{m\to n}|}
\sum_{i\in Q_{m\to n}}
\log
\frac{\exp(u_i^m\cdot P^M[n,y_i]/\tau_P)}
{\sum_{z\in\mathcal V_n}\exp(u_i^m\cdot P^M[n,z]/\tau_P)}.
$$

其中：

- $u_i^m$ 是当前 Query 图像的 L2 pre-BN CLS；
- $u_i^m\cdot P^M[n,z]$ 是 Query 与目标模态身份原型的余弦相似度；
- $\tau_P=0.1$；
- $\mathcal V_n$ 是目标模态中已初始化的身份原型集合；
- $Q_{m\to n}$ 仅包含其真实身份在目标模态中具有有效原型的 Query。

六个方向为 RGB→NIR、NIR→RGB、RGB→TIR、TIR→RGB、NIR→TIR、TIR→NIR。每个方向先对有效 Query 求均值，再对有效方向求均值，避免某个方向因有效样本数量较多而自然获得更大权重。

### 4.5 TIR→TIR 轻量辅助项

为显式覆盖 TIR 模态内身份原型判别，增加：

$$
\mathcal L^P_{T\to T}
=-\frac1{|Q_{T\to T}|}
\sum_i
\log
\frac{\exp(u_i^T\cdot P^M[T,y_i]/\tau_P)}
{\sum_{z\in\mathcal V_T}\exp(u_i^T\cdot P^M[T,z]/\tau_P)}.
$$

定义：

$$
\mathcal L_{MP}
=\operatorname{Mean}_{m\ne n}\mathcal L^P_{m\to n}
+0.1\mathcal L^P_{T\to T}.
$$

若某方向没有有效 Query，则跳过该方向；不能计算全屏蔽 softmax，也不能让无效原型作为负类参与分类。

### 4.6 MP 梯度

MP 的梯度从当前 Query 特征 $u_i$ 回传到对应的 pre-BN CLS，再进入共享 ViT。目标原型 $P^M$ 是 stop-gradient 记忆，不接收当前反向梯度。原型仅在 optimizer 成功更新后，使用当前 batch 的 detached 中心 EMA 更新。

---

## 5. 新增模块二：Pair/Mixed 图像级对比

### 5.1 设计目标

原 UAD WRT 已经使用真实图像正负距离，但没有显式将每个有向目标模态候选集合拆开组织。Pair/Mixed 进一步直接在真实图像特征之间建立目标模态检索约束：

- **Pair：** 指定源模态 Query 对指定目标模态 Gallery；
- **Mixed：** 指定源模态 Query 对 RGB、NIR、TIR 共同构成的混合 Gallery。

Query 是当前 batch 中真实图像的 L2 pre-BN CLS，不是可学习 Query token，也不需要初始化额外网络参数。

### 5.2 相似度矩阵

对源模态 $m$ 与目标模态 $n$：

$$
S^{m,n}_{ij}=\frac{u_i^m\cdot u_j^n}{\tau_C},
\qquad \tau_C=0.1.
$$

所有新损失的相似度矩阵、masked log-sum-exp 和 softmax 在 FP32 下计算。模型前向仍可使用原 UAD 的 AMP。

### 5.3 相机与有效候选掩码

遵循当前实际使用的 UAD evaluator：若 Gallery camera ID 与 Query camera ID 相同，则该候选不参与训练对比。基本有效掩码为：

$$
V_{ij}=\mathbf 1[c_i\ne c_j].
$$

对 Mixed，同一张图像可能同时作为 Query 和 Gallery 候选，必须额外屏蔽该图像的自匹配对；自匹配屏蔽独立于 camera mask。若不同数据加载路径可能重复同一路径，应使用稳定样本 ID 或路径映射识别相同图像，不能只依赖对角线位置。

有效正样本：

$$
P_{ij}=V_{ij}\mathbf 1[y_i=y_j].
$$

有效负样本：

$$
N_{ij}=V_{ij}\mathbf 1[y_i\ne y_j].
$$

只有同时至少包含一个有效正样本和一个有效负样本的 Query 才参与该行损失。不能只屏蔽无效正样本而让无效负样本留在 softmax 分母中。全无效行应在进入 log-softmax 前剔除，防止 NaN。

### 5.4 Uniform 与 Set 两种正样本目标

设一行有效候选集合为 $\mathcal V_i$，有效正样本集合为 $\mathcal P_i$。

Uniform 多正样本损失：

$$
\mathcal L_{uni}^{(i)}
=-\frac1{|\mathcal P_i|}
\sum_{j\in\mathcal P_i}
\log
\frac{\exp(S_{ij})}
{\sum_{k\in\mathcal V_i}\exp(S_{ik})}.
$$

该项借鉴 MINER 的多正样本目标，让多个有效正样本都得到关注；但它会约束正样本之间的概率分布，不能理解成只要求正确身份总概率变大。

Set 损失：

$$
\mathcal L_{set}^{(i)}
=-\log
\frac{\sum_{j\in\mathcal P_i}\exp(S_{ij})}
{\sum_{k\in\mathcal V_i}\exp(S_{ik})}.
$$

Set 目标关注正确身份图像集合的总概率，不强制正样本概率均分。

A-V1 固定混合：

$$
\mathcal L_{img}^{(i)}
=0.3\mathcal L_{uni}^{(i)}
+0.7\mathcal L_{set}^{(i)}.
$$

该混合既保留多正样本覆盖，又让主要梯度目标落在正确身份集合战胜错误候选上。它不是 mAP、Rank-1 或 mINP 的直接可微优化公式。

### 5.5 Pair 与 Mixed

Pair 对六个有向模态对分别计算行损失：

$$
\mathcal L_{pair}
=\operatorname{Mean}_{m\ne n}
\mathcal L_{img}(U_m,U_n).
$$

Mixed 对每个源模态构造 Gallery：

$$
U_{mix}=[U_R;U_N;U_T],
$$

并分别计算 RGB→Mixed、NIR→Mixed、TIR→Mixed：

$$
\mathcal L_{mixed}
=\operatorname{Mean}_{m\in\{R,N,T\}}
\mathcal L_{img}(U_m,U_{mix}).
$$

Pair 内部六个方向等权；Mixed 内部三个源模态等权。最终图像级损失为：

$$
\mathcal L_{img,total}
=0.75\mathcal L_{pair}+0.25\mathcal L_{mixed}.
$$

Pair 与 Mixed 的定义、有效 Query 规则和平均方式必须固定，不再在首轮增加更多子损失。

### 5.6 图像级对比的梯度

Pair/Mixed 的梯度同时通过 Query 端与 Gallery 端特征回传。一个图像作为 Query 时接收本行 softmax 梯度，作为 Gallery 候选时也接收其他 Query 的梯度。所有梯度最终汇入共享 ViT 的同一个 CLS 表征，不存在单独的 Query/Gallery 网络。

---

## 6. 新增模块三：软困难 PID 采样

### 6.1 参考 MINER 的范围

MINER 公开代码中的软困难 PID 选择思想是：利用预计算身份中心的相似度，提高相似异身份共同进入 batch 的概率。A-V1 只迁移这一思想，不迁移 MINER 的完整 Q/G 训练组织、方差控制器或最远相机对策略。

### 6.2 为什么不从 ImageNet 特征直接启用困难采样

ImageNet 预训练特征尚未针对 WHU 训练身份形成可靠的 ReID 判别结构。若直接用它们决定“最困难身份”，容易把低质量的初始相似度当作可靠困难度。

因此 A1 的 PID batch schedule 在前 10 个 Epoch 保持原 UAD PKM 随机调度，但从 Epoch 1 起使用相机多样性优先的 K-group 重分组。模型完成前 10 个 Epoch 后，在 Epoch 10 结束处用当前 A1 模型对完整训练集执行一次确定性特征提取，构造并落盘固定身份困难度参考；从 Epoch 11 起只通过受约束的跨 batch group 交换提高困难身份共现。该参考随后固定，不在每个 iteration 或 epoch 在线刷新。

### 6.3 困难度

对身份 $y$ 计算当前模型的全局身份中心 $\bar p_y^G$，并计算有效的三模态身份中心 $\bar p_{y,R},\bar p_{y,N},\bar p_{y,T}$。所有中心先对每张图像特征 L2，再求身份均值并 L2。

全局困难度：

$$
h_G(y,z)=\cos(\bar p_y^G,\bar p_z^G).
$$

TIR 相关困难度 $h_T(y,z)$ 为身份 y 与身份 z 之间所有有效的 TIR↔TIR、TIR↔RGB、TIR↔NIR、RGB↔TIR、NIR↔TIR 身份中心余弦相似度的平均；缺失模态组合不参与平均。计算时同时纳入身份对的两个顺序，最终将困难度矩阵显式对称化为 $h=(h+h^T)/2$，避免交换评分依赖身份顺序。

$$
h(y,z)=0.5h_G(y,z)+0.5h_T(y,z),\quad y\ne z.
$$

这是 A-V1 针对 WHU 任务的适配定义，不是 MINER 原论文的逐字公式。

### 6.4 先固定原 PKM 的 epoch 账本，再进行受约束的困难共现交换

不能在每个 batch 直接从全体可用 PID 中重新抽取“8 个随机 + 8 个困难”并声称只改变共现关系。这样的在线重采样可能让少数困难 PID 提前耗尽 K-group，使剩余 group 无法组成完整 batch，进而改变 epoch 步数、每个 PID 的曝光次数和样本消耗。

A1 使用两阶段、可核验的 epoch schedule：

**阶段 A：物化原 PKM 基准 schedule。**

1. 按原 UAD PKM 的组池生成、PID 可用性判断、每 batch PID 数、模态顺序、每 PID 每模态 K 张图像、耗尽规则和尾部处理规则，生成该 rank 在本 epoch 原本会执行的完整 batch schedule。
2. 每条 PID-group 记录包含一个 PID 以及该 PID 在 RGB/NIR/TIR 三个模态对应的 K 张图像索引。保留原 sampler 实际会消耗的记录；原算法本来会留下未使用 group 的话，也保持其原有处理方式，不额外补齐或删除。
3. 记录基准 schedule 的 batch 数、每个 PID 的出现次数、每个 PID 各模态 group 数，以及每个样本索引在本 epoch 被使用的 occurrence 次数。这个账本是之后交换的硬约束。

**阶段 B：仅通过交换现有 group 记录来增加困难共现。**

1. Epoch 1–10 不做困难交换，PID batch schedule 与原 PKM 的随机 schedule 一致。
2. Epoch 11–120，从固定困难参考得到身份间相似度 `h(y,z)`。以每个 batch 中的 PID 集合 `B` 定义困难度分数：

$$
H(B)=\frac{1}{|\{(y,z):y,z\in B,y<z\}|}\sum_{y<z,\,y,z\in B}h(y,z).
$$

3. 每个 epoch 将约 50% 的 group 记录位置作为“困难化候选位置”，随机打乱候选位置顺序。对一个候选记录 `y`，从其他 batch 的一小组候选记录 `z` 中寻找交换对象。只允许在交换后两个 batch 内 PID 仍然互不重复的情况下交换整个 PID-group 记录。
4. 对每个可行交换，计算两个 batch 的总困难度变化：

$$
\Delta=H(B_a^{swap})+H(B_b^{swap})-H(B_a)-H(B_b).
$$

仅从 `Δ>0` 的可行交换中，按 `softmax(Δ/τ_s)` 概率选择一个交换；若没有可行的正增益交换，则保留原位置。每个 group 记录在本轮困难化阶段最多参与一次交换，避免反复交换导致调度不稳定。候选池可固定为每个目标位置最多随机检查 32 条其他 batch 记录，避免全量两两搜索。
5. 每次交换只把两个现有 PID-group 记录互换位置，不复制、不丢弃、不重新抽取样本。由此可严格保持：epoch batch 数不变、每个 batch 的 PID 数不变、batch 内 PID 不重复、每个 PID 的 epoch 出现次数不变、每个 PID-group occurrence 不变、样本 occurrence 多重集不变、每个 rank 的训练步数不变。

`hardening_slot_fraction=0.5` 表示最多约一半的记录位置会尝试困难化，并不承诺每个 batch 恰好一半 PID 是由困难抽样生成。这个定义优先保证原 PKM 的样本账本与训练长度，而不是为了达到名义 1:1 比例破坏样本消耗规律。

### 6.5 DDP 困难参考的提取、保存与恢复

Epoch 10 结束后，所有 rank 先完成该 epoch 的训练与原型更新并同步。随后：

1. rank 0 使用当前 A1 模型的未包装实例 `model.module`（不能调用只有 rank 0 参与的 DDP wrapper forward），在完整训练划分上进行一次确定性、无随机增强、顺序遍历、`drop_last=False` 的特征提取。非 rank 0 等待，不自行只提取本地分片。
2. 计算全局身份中心、RGB/NIR/TIR 模态身份中心、涉及 TIR 的相似度和最终困难度矩阵。无有效模态对时跳过该项；无有效 TIR 项时退回有效全局困难度；任何 NaN/Inf 都必须在保存前处理并断言通过。
3. 在提取前先保存 Epoch 10 结束时的模型状态（至少保留 model state，并记录其哈希），然后保存完整参考 artifact，例如 `hard_reference_epoch010.pt`。artifact 至少包括：全局/模态身份中心、困难度矩阵、训练划分与样本 manifest 哈希、PID 连续映射、模态/相机元数据映射、预处理配置、用于提取特征的 Epoch 10 模型 state 哈希、生成 epoch、`τ_s` 和构建代码版本标识。保存文件后记录其 SHA-256；最终 Epoch 10 训练 checkpoint 同时记录该 artifact 的路径与哈希。
4. rank 0 将 artifact 路径、哈希与困难度张量广播给所有 rank，或由所有 rank 从共享路径加载同一个 artifact 并核验哈希。barrier 后统一从 Epoch 11 开始困难化交换。每个 rank 必须使用同一份参考。
5. 该参考在本次完整训练中固定，不逐 iteration 或逐 epoch 刷新。断点恢复时必须加载原 artifact 并校验哈希，不能悄悄用恢复时的当前模型重新计算，否则训练轨迹就发生改变。如果 artifact 丢失，只允许从保存的 Epoch 10 模型状态按完全相同的确定性流程显式重建，并在日志中标记重建；无法确认同一状态时应 fail-fast。
6. 断点 checkpoint 保存参考 artifact 的路径与哈希；共享文件系统不可用时，将必要的中心/困难度张量一并纳入 checkpoint 或使用受支持的分布式广播方案。

离线提取只使用训练划分，不涉及测试图像或标签。为了避免改变训练随机序列，应使用独立的 DataLoader generator，并在同一进程中执行时保存/恢复 RNG 状态。

### 6.6 相机多样性优先的 K-group 构造

相机过滤会把同相机 Query–Gallery 候选排除。如果原 PKM 为同一 PID、同一模态抽到的 K 张图像大多来自同一相机，那么 Pair/Mixed 可能有大量 Query 没有有效正样本。A1 因此在 K-group 构造阶段增加轻量相机多样性优先策略，但不新增跨平台配对约束，也不改变损失的相机掩码。

具体约束和步骤：

1. **先保留原 PKM 的样本 occurrence 账本。** 先由原 PKM 生成该 epoch 实际会消耗的 K-group schedule。对每个 PID、每个模态，收集这些已消耗 group 中的所有图像 occurrence；同一图像若因原始采样规则被重复使用，就保留对应的重复 occurrence。不得额外复制、丢弃或替换图像索引。
2. **只重分组，不改 occurrence 多重集。** 对同一 PID、同一模态的 occurrence 列表，保留总 occurrence 数与 K-group 数不变，将 occurrence 重新分配到同样数量的 K 大小组中。每个新 group 构造时，优先从尚未出现在该组的 camera ID 中随机选择一张；若剩余候选已无法提供新的 camera ID，则从所有剩余 occurrence 中随机补足。随机并列选择由 `seed + epoch + rank + PID + modality` 派生的独立 RNG 控制。
3. **处理相机不足。** 如果该 PID/模态只有一个相机，或剩余 occurrence 都来自已出现相机，就允许重复相机；不人为复制新图像、不丢弃样本，也不要求不同平台配对。
4. **按原 PID schedule 回填。** 将重新分组得到的 K-group 依次回填到该 PID 在原 schedule 中的 group 槽位，然后再执行第 6.4 节的跨 batch 受约束交换。每个 group 仍然是同一个 PID 的 RGB/NIR/TIR 三模态组，保持原来的模态拼接顺序。
5. **影响范围透明记录。** 该操作保持每个 PID、每种模态的样本 occurrence 频次和 K-group 数不变，但会改变哪些图像在同一个 K-group 内共同形成 ProCA 模态中心。这是 A1 明确采用的相机多样性适配，而不是声称完全复刻原始 PKM 的组内组合。ProCA 的公式、归一化和损失权重本身不改。
6. **验证实际收益条件。** 记录六个 Pair 方向、三个 Mixed 方向的有效 Query 数及比例。相机多样性只提高有效跨相机正样本出现机会，不保证每个 Query 都有正样本；仍按既定规则跳过没有有效正样本或有效负样本的 Query。

## 7. 完整训练计划与参数

### 7.1 初始化与训练起点

A1 和 B0 均从同一个 ImageNet 预训练 ViT-B/16 权重开始。使用 UAD 原始的预训练加载逻辑和模型构造方式，分类器按原 UAD 规则针对当前训练身份数初始化。不得加载已收敛的 UAD ReID checkpoint，也不得加载旧 ReID 分类器来替代 ImageNet 起点。

官方 UAD 配置以单次完整 120 Epoch 训练为基准，使用 SGD、5 Epoch 学习率 warm-up、原 UAD 的学习率调度和数据增强。官方 `configs/UAD.yml` 默认值包含 `MAX_EPOCHS=120`、`PRETRAIN_CHOICE=imagenet`、`BASE_LR=0.008`、`IMS_PER_BATCH=64`、`WARMUP_EPOCHS=5`、`WEIGHT_DECAY=1e-4`、`BIAS_LR_FACTOR=2`。官方 README 的四卡示例显式覆盖为 `IMS_PER_BATCH=256`、`BASE_LR=0.032`。实验必须选用一套与实际 GPU/复现基线相符的完整配置，B0/A1 逐项一致，不能混搭不同配置的 batch 与 LR。

### 7.2 A1 新损失权重

$$
\lambda_P(t)=0.1r(t),\qquad
\lambda_C(t)=0.1r(t),\qquad
r(t)=\min(t/5,1),
$$

其中 Epoch $t$ 从 1 开始。前 5 Epoch 新增损失线性 warm-up；原 UAD 学习率 warm-up 独立沿用，不替换或重写。

| 参数 | A-V1 首轮值 |
|---|---:|
| Backbone | UAD 原 ViT-B/16 |
| 训练起点 | ImageNet 预训练权重 |
| 完整训练轮数 | 120 Epoch（沿用所选原 UAD 完整配置） |
| 原 ProCA 权重 | 0.01 |
| 原 GPD 温度 | 0.03 |
| 原 GPD EMA 新中心系数 | 0.2 |
| MP 温度 $\tau_P$ | 0.1 |
| MP 权重 $\lambda_P$ | 0.1，前 5 Epoch warm-up |
| TIR→TIR 内部权重 | 0.1 |
| 图像对比温度 $\tau_C$ | 0.1 |
| 图像损失权重 $\lambda_C$ | 0.1，前 5 Epoch warm-up |
| Uniform/Set 比例 | 0.3 / 0.7 |
| Pair/Mixed 比例 | 0.75 / 0.25 |
| A1 K-group 构造 | 全程相机多样性优先重分组，occurrence 多重集不变 |
| 软困难共现交换 | Epoch 1–10 不交换；Epoch 11–120 最多约 50% group 记录位置尝试困难化 |
| 困难交换候选池 | 每个目标位置最多随机检查 32 条其他 batch 记录 |
| 困难采样温度 $\tau_s$ | 0.2 |
| 推理描述子 | 768D pre-BN CLS，L2 |
| Rerank/TTA | 关闭 |

### 7.3 B0 与 A1

**B0：** 完整 UAD 基线训练。ImageNet 初始化、原始 PKM 组内构造与 PID 调度、原 CE/WRT/ProCA/GPD、原优化器与调度，完整 120 Epoch；GPD 的原型/EMA 规则保留，并使用与 A1 相同的 AMP 成功步更新门控。

**A1：** 同一 ImageNet 初始化、同一原始训练配置和预算；保留四项原损失，增加 MP、Pair/Mixed、相机多样性 K-group 重分组与软困难共现交换。MP Bank 使用 bootstrap 初始化；GPD Bank 仍按原 UAD 方式初始化，公式/EMA 不变，并使用与 B0 相同的 AMP 成功步门控。

两组使用相同训练划分、图像尺寸、增强、类别映射、总训练 Epoch、优化器、学习率调度、评测协议和尽可能一致的随机种子。A1 从 Epoch 1 起使用相机多样性优先的 K-group 重分组，但 Epoch 1–10 的 PID batch schedule 仍沿用原 PKM 随机调度；Epoch 11 起才在固定参考上执行受约束的困难共现交换。B0 全程使用原 UAD PKM 的原始组内构造与 PID 调度。

### 7.4 评估协议

- `NECK_FEAT=before`；
- `FEAT_NORM=yes`；
- 评测特征为 768D pre-BN CLS；
- 欧氏距离；
- 严格使用当前官方/本地正式 evaluator 的同相机过滤；
- 不使用 rerank、TTA 或测试时模态标签；
- 固定 Query/Gallery 划分；
- 输出总体五项指标和九种有向模态组合结果。

---

## 8. 正向数据流、梯度流与 Bank 更新时序

```text
ImageNet 预训练 ViT-B/16
            │
            ├─ A1 训练前：rank 0 完整训练集确定性提特征
            │      └─ 初始化 MP Bank/valid → 广播并保存
            │
            ▼
WHU 训练图像（RGB / NIR / TIR）
            │
   原 PKM epoch 账本与 K-group 数
            │
   A1：相机多样性重分组（不改 occurrence 多重集）
            │
   Epoch 1–10：原随机 PID schedule
   Epoch 11+：受约束的跨 batch group 交换
            │
            ▼
       共享 ViT-B/16
            │
       Final LayerNorm
            │
            ▼
       pre-BN CLS f [3B,768]
            │
      ┌─────┴──────────────────────────┐
      │                                │
      ▼                                ▼
 BNNeck + Classifier                 L2(f)
      │                                │
      ▼                  ┌─────────────┼───────────────┐
     CE                  ▼             ▼               ▼
                       原 GPD         MP           Pair/Mixed
                         │             │               │
        原 WRT / ProCA ──┴─────────────┴───────────────┘
                         │
                         ▼
                    合并总损失
                         │
                         ▼
                    反向传播
                         │
                         ▼
                 unscale / optimizer step
                         │
                 检查 AMP 是否成功更新
                    ┌────┴────┐
                    │成功     │跳过
                    ▼         ▼
              DDP 聚合中心   不更新任何 Bank
                    │
            ┌───────┴────────┐
            ▼                ▼
         GPD Bank          MP Bank
      原 EMA 公式       独立 EMA 公式

推理：图像 → 共享 ViT → pre-BN CLS → L2 → 欧氏距离排序
```

### 8.1 反向梯度路径

- **CE：** `cls_score` 的分类误差经过 Classifier、BNNeck，回传到 pre-BN CLS 与共享 ViT。
- **WRT：** 原始 WRT 的正负距离目标直接对其输入的 pre-BN 特征产生梯度，再回传至共享 ViT。
- **ProCA：** 当前模态中心接收梯度；统一身份中心按原实现 detach，梯度不会穿过该中心分支回传。
- **GPD：** 当前图像特征接收梯度；历史 GPD Bank 是无梯度记忆，不接收反向梯度。
- **MP：** 当前 Query 的 L2 特征接收梯度；目标模态 Bank、有效性掩码和标签均无梯度。
- **Pair/Mixed：** Query 与 Gallery 两端的 L2 特征都接收梯度。同一图像可同时作为 Query 与其他 Query 的 Gallery 候选，其梯度在同一共享 CLS 上累加。
- **采样与相机掩码：** PID schedule、相机分组、困难度参考、样本索引、mask 和所有 Bank 均不接收梯度。

所有损失的梯度最终在共享 ViT 参数上累加；A-V1 不存在单独的 Query/Gallery 编码器或额外推理分支。

### 8.2 AMP 成功步与 Bank 更新规则

GPD 的原型定义、中心聚合、EMA 系数和更新公式保持原 UAD 逻辑；但为避免 AMP 溢出导致“参数没更新、记忆却前进”，B0 与 A1 共同使用同一个成功步门控：

1. 前向时用更新前的 Bank 计算损失，并缓存当前 batch 的 detached 身份中心。
2. 执行 `scaler.unscale_(optimizer)`（如启用 AMP）及原配置中的梯度裁剪。
3. AMP 启用时，在 `scaler.step(optimizer)` 前记录 `old_scale=scaler.get_scale()`，随后调用 `scaler.step(optimizer)` 与 `scaler.update()`，再读取 `new_scale`。不能使用 `scaler.step()` 的返回值判断是否成功，因为普通 optimizer 的 `step()` 通常返回 `None`。
4. `new_scale < old_scale` 视为本步因非有限梯度而跳过；`new_scale >= old_scale` 视为本步成功。AMP 未启用且 `optimizer.step()` 正常完成时视为成功。
5. DDP 下各 rank 对成功标志执行一致性检查（例如 all-reduce MIN/MAX）。若各 rank 判断不一致，所有 rank 都不得更新 Bank，并应 fail-fast 以免记忆分叉。
6. **仅成功步更新 Bank。** 成功时先跨 rank 聚合同一 `(PID, modality)` 的重复中心，再按原 GPD 更新逻辑更新 GPD Bank，并按 A-V1 的 EMA 更新 MP Bank；跳过步不更新任何 Bank。B0 也采用相同的 GPD 成功步门控，以保证比较条件一致。

这是一项共同的数值稳定性与状态一致性处理，不改变 GPD 的损失定义或 EMA 数值；在没有 AMP 溢出时，其行为与原有更新顺序一致。日志必须记录跳过步次数与 Bank 更新次数。

### 8.3 新增损失的早期梯度贡献检查

只记录 loss 标量不足以证明 MP、Pair/Mixed 真正影响了共享 CLS。正式训练前和训练早期进行少量诊断，不构建长期诊断系统：

- Smoke Test 至少选取一个确实有有效 Query 的 batch；训练早期再分别用 Epoch 1（新增权重 warm-up 中）和 Epoch 5（新增权重达到 0.1 后）的模型状态做一次诊断。
- 使用与该模型状态对应的单进程诊断副本、保存的确定性 batch、当时的 GPD/MP Bank 与 valid mask 快照、相机/PID 元数据和对应的新增损失权重，避免在主 DDP 训练循环中引入额外 backward 或 collective。分别计算加权原始损失 `L_base=CE+WRT+0.01·ProCA+GPD`、`λP·L_MP`、`λC·0.75·L_pair`、`λC·0.25·L_mixed` 对 `global_feat` 的梯度范数。
- 使用 `torch.autograd.grad(loss_component, global_feat, retain_graph=True, allow_unused=True)` 获取中间特征梯度。该操作在独立诊断副本中执行；主训练仍只对总损失做一次正常 backward。诊断不执行 optimizer step，不更新 Bank，也不改变正式训练的参数或随机状态。
- 每项记录原始 loss、加权 loss、梯度范数、梯度是否有限、相对于 `L_base` 的范数比，以及各方向有效 Query 数/比例。MP、Pair、Mixed 应有有限且非零的梯度；若梯度为零，先检查有效 Query、mask、Bank valid 与特征连接。
- 若新增项相对基础损失的梯度贡献持续接近数值零，或相差多个数量级，应先复核有效样本与归一化/权重接线；这是一条排错提示，不是需要通过调参强行达到的固定比例。梯度范数只反映共享特征接口处的局部贡献，不能替代最终检索指标。

诊断仅在少量 batch 上执行；正式训练不需要每步额外计算多次梯度。

## 9. 关键风险与应对原则

| 风险 | 处理原则 |
|---|---|
| ImageNet 初始特征身份判别较弱 | MP 权重前 5 Epoch warm-up，MP Bank 随训练 EMA 更新 |
| 初始困难度不可靠 | 前 10 Epoch 随机 PKM，使用已训练 10 Epoch 的 A1 模型建立固定困难参考 |
| MP 与 GPD 有功能重叠 | 保留 GPD 原损失与权重，MP 只补充目标模态身份竞争 |
| Pair/Mixed 与 WRT 有部分重叠 | 保留 WRT 原定义，新增对比作为补充而非替代 |
| Uniform 目标可能约束正样本内部概率 | Uniform 权重固定为 0.3，Set 权重为 0.7 |
| 同相机掩码造成有效正样本不足 | 统计每个方向有效 Query 比例；无有效正负样本的 Query 跳过 |
| 困难采样改变身份频率 | 先物化原 PKM schedule，再只交换现有 group 记录；严格保留 PID 出现次数、样本 occurrence 多重集、batch 数和 DDP 步数 |
| 相机正样本不足 | A1 在同 PID/模态内优先把现有 occurrence 分到相机更多样的 K-group，并记录有效 Query 比例 |
| 新增监督损害原有特征几何 | 保留所有原损失，按固定 warm-up 启用新增项，检查各项梯度范数与检索指标 |
| 训练损失下降但检索无提升 | 以正式检索指标为准，不把损失下降等同于方法有效 |

---

## 10. 参考论文与实现边界

### UAD / WHU-MARS

- 论文与数据集：<https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_WHU-MARS_A_Multispectral_Aerial-Ground_Benchmark_Towards_Any-Scenario_Person_Re-Identification_CVPR_2026_paper.pdf>
- 官方仓库：<https://github.com/msm8976/WHU-MARS>
- UAD 配置：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/configs/UAD.yml>
- 训练说明：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/README.md>
- ProCA/GPD/训练循环：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/processor/processor.py>
- 模型、BNNeck、Classifier：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/model/make_model.py>
- PKM/DDP sampler：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/datasets/sampler_ddp.py>
- WRT：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/loss/wrt_loss.py>
- 评测与相机过滤：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/utils/metrics.py>

### MINER

- 官方仓库：<https://github.com/oronnir/MINER>
- 软困难 PID 采样：<https://github.com/oronnir/MINER/blob/main/miner_tvrid_rgb_final/sampler.py>
- 多正样本损失：<https://github.com/oronnir/MINER/blob/main/miner_tvrid_rgb_final/losses.py>
- 训练流程：<https://github.com/oronnir/MINER/blob/main/miner_tvrid_rgb_final/train.py>

**来源边界：** UAD 的骨干、原损失定义、ProCA/GPD 公式与 EMA、原始 PKM 和评测协议来自原论文/公开代码。B0/A1 共同采用的 AMP 成功步 Bank 更新门控是训练状态一致性的工程处理，不改变 GPD 公式。MINER 提供软困难身份选择与多正样本对比的参考。三模态 MP Bank、六方向目标模态分类、TIR→TIR 辅助项、Pair/Mixed 组织、Uniform/Set 混合、WHU 相机掩码、相机多样性 K-group 重分组、受约束跨 batch group 交换，以及 Epoch 10 后固定困难参考，均为 A-V1 的适配设计，需要通过完整训练验证其实际效果。
