# A-V1 代码修改指导：基于原 UAD 的完整训练实现

> **目标：** 在官方 UAD 工程中实现 A-V1，并从 ImageNet 预训练 ViT-B/16 开始完成原 UAD 的完整训练预算。  
> **训练起点：** ImageNet 预训练权重；不加载已收敛的 UAD ReID checkpoint。  
> **实现原则：** 尽量沿用原 UAD 的模型、损失、训练入口、优化器、学习率调度、PKM 和评测器，只新增 MP、Pair/Mixed、软困难 PID 采样及必要的 Bank 管理。

---

## 0. 实现前冻结的决定

1. 完整执行所选原 UAD 训练配置，基准为官方单次 120 Epoch 训练。
2. A1 与 B0 都从同一个 ImageNet 预训练 ViT-B/16 初始化。
3. 原 UAD 的 CE、WRT、ProCA、GPD 损失定义、特征输入和权重不改；训练循环只为 B0/A1 共用地增加 AMP 成功步 Bank 更新门控。
4. GPD Bank 的原型定义、中心聚合、EMA 公式与系数保持原 UAD 逻辑；B0/A1 共同增加 AMP 成功步门控，只有 optimizer 真正更新参数后才更新 GPD Bank。
5. A1 新增 MP Bank，训练前通过 ImageNet 初始化模型的无梯度训练集特征 pass 初始化；训练过程中 EMA 更新。
6. 前 5 Epoch 对 MP 与图像对比损失做权重 warm-up；原 UAD 学习率 warm-up 和 scheduler 原样保留。
7. A1 从 Epoch 1 起对原 PKM 实际会消耗的样本 occurrence 做相机多样性优先重分组；Epoch 1–10 的 PID batch schedule 保持原随机调度。Epoch 10 结束后生成并保存固定困难参考；Epoch 11 起对现有 PID-group 记录做受约束跨 batch 交换，以提高困难身份共现，同时严格保留每个 PID 的出现次数、样本 occurrence 多重集、batch 数和 DDP 步数。
8. Pair/Mixed 的 Query 都是当前图像 pre-BN CLS 的 L2 特征，不是可学习 Query token。
9. 训练与测试的相机过滤必须与当前正式 evaluator 保持一致。
10. B0 与 A1 使用同一完整训练预算、原始 UAD 配置、模型起点和评测协议。

---

## 1. 原始代码阅读与修改边界

实施前先阅读本地实际执行版本，不依据文件名猜测接口。重点位置：

| 位置 | 阅读目的 | A-V1 处理 |
|---|---|---|
| `configs/UAD.yml` 与启动命令 | 训练轮数、batch、LR、预训练、输入尺寸、ProCA/GPD 参数 | 作为完整训练配置依据 |
| `model/make_model.py` | backbone、Final LN、CLS、BNNeck、Classifier、返回值 | 保留模型结构和返回契约 |
| `model/backbones/vit_pytorch.py` | ImageNet 权重加载、ViT 特征流 | 保留原预训练加载 |
| `processor/processor.py` | 训练循环、CE/WRT/ProCA/GPD、AMP、Bank 更新 | 在原循环中接入新损失 |
| `loss/wrt_loss.py` | WRT 距离与权重计算 | 不改 |
| `datasets/sampler.py` / `sampler_ddp.py` | PKM 分组、三模态顺序、DDP 切分 | 物化原 schedule；A1 增加相机多样性重分组与受约束 group 交换，并断言样本账本不变 |
| `utils/metrics.py` | 特征提取、L2、同相机过滤、九组合评测 | 不改推理定义 |
| `solver/make_optimizer.py` 等 | SGD、参数组、LR、scheduler | 保留原完整训练配置 |

建议新增独立的 MP/Pair/Mixed 损失模块、Bank 初始化/更新辅助函数、PKM schedule 物化与账本断言、相机多样性 K-group 重分组、受约束跨 batch group 交换、配置项和测试；不建议复制整个训练框架。

---

## 2. 完整训练初始化

### 2.1 Backbone 与分类器

A1/B0 都使用原 UAD 的 ImageNet 预训练加载路径：

- 加载 ImageNet ViT-B/16 backbone 参数；
- 按原 UAD 模型构造流程创建 BNNeck；
- 按当前 WHU 训练身份数创建并初始化 Classifier；
- 使用原 UAD 的权重初始化方式；
- 不加载任何已收敛的 ReID checkpoint；
- 不通过 `load_param()` 恢复旧 ReID 分类器；
- 记录 ImageNet 权重文件路径和哈希、配置文件哈希、类别映射和随机种子。

原 UAD 普通 ReID 权重加载函数可能跳过 `classifier`，但当前任务不需要加载 ReID 分类器：分类器应按完整训练的原 UAD 方式初始化，而不是从不存在的 ReID checkpoint 恢复。

### 2.2 训练配置的唯一来源

官方 `configs/UAD.yml` 默认包括 `MAX_EPOCHS=120`、`PRETRAIN_CHOICE=imagenet`、`BASE_LR=0.008`、`IMS_PER_BATCH=64`、`WARMUP_EPOCHS=5`、`WEIGHT_DECAY=1e-4`、`BIAS_LR_FACTOR=2`。官方 README 的四卡命令显式覆写为全局 `IMS_PER_BATCH=256`、`BASE_LR=0.032`。

具体实验必须使用与当前 UAD 完整基线相同的一套启动命令和配置。不能将默认 batch 与四卡 LR 混合。B0/A1 的训练轮数、batch、优化器、scheduler、数据增强和 DDP 配置必须逐项一致。

### 2.3 MP Bank bootstrap pass：完整训练集、单一权威提取、广播与落盘

新增模态 Bank 需要在训练开始时有可用原型。bootstrap 必须覆盖完整训练划分，不能让 rank 0 只提取自己的分片，也不能让 rank 0 单独调用 DDP wrapper forward。

1. **时机：** 完成模型 ImageNet 权重加载与类别映射后、正式训练第一个 optimizer step 前执行。建议在模型尚未包装为 DDP 时执行；若工程必须先包装，则 rank 0 只调用 `model.module`。
2. **数据：** 创建独立的确定性训练集视图，使用与评测一致的确定性 resize/normalize，不使用训练随机增强。采用顺序遍历、`drop_last=False`，不使用 `DistributedSampler`，确保完整训练划分每张图像都被遍历。仅允许训练划分，不读取测试 Query/Gallery。
3. **执行：** 仅 rank 0 进行特征提取，模型设为 `eval()`，使用 `torch.inference_mode()`；其他 rank 等待。保存并恢复模型模式、Python/NumPy/Torch CPU/CUDA RNG 状态；DataLoader 使用独立 generator，避免离线 pass 改变训练的随机序列。不得调用 DDP 包装模型 `model(...)`，否则其他 rank 不参与时可能触发同步问题。
4. **聚合：** 提取 `global_feat`（768D pre-BN CLS）。每张图像先 L2，再按 `(连续 PID, modality)` 聚合均值，并对均值 L2。写入 `MP_Bank[modality, PID]` 与 `MP_valid[modality, PID]`。没有样本的组合保持 invalid。
5. **保存与广播：** rank 0 保存 Bank、valid mask、训练划分/manifest 哈希、连续 PID 映射、预处理配置、ImageNet 权重哈希、生成时间/版本与 Bank 哈希。随后将 Bank 和 valid mask 广播给其他 rank，或在共享文件系统上让所有 rank 读取同一 artifact 并核验哈希。所有 rank barrier 后才进入训练。
6. **不改变训练起点：** bootstrap 不执行 backward、不调用 optimizer、不更新 backbone/BNNeck/classifier，也不推进 scheduler。它只初始化新增 Bank。原 GPD Bank 仍按原 UAD 初始化方式处理，不用 bootstrap 特征重写 GPD Bank；其原型公式/EMA 不变，但更新受 B0/A1 共用的 AMP 成功步门控约束。
7. **断点恢复：** 正式训练 checkpoint 必须保存 MP Bank/valid 与其版本信息。正常恢复时直接恢复 Bank，不重新执行 ImageNet bootstrap；只有全新训练启动才执行此 pass。

初始 MP 原型来自 ImageNet 特征，可能不具备稳定的 WHU 身份判别性，因此新增 MP 权重仍按前 5 Epoch 的预定 warm-up 启用。

## 3. 特征与元数据契约

### 3.1 前向返回值

原 UAD forward 的返回值应保持：

- `cls_score`：BNNeck 后特征经 Classifier 的 logits；
- `global_feat`：pre-BN CLS，维度 768；
- `feat`：BNNeck 后特征。

模型输入拼接顺序保持 `[RGB; NIR; TIR]`。如果每模态 batch 大小为 B，则输出 `global_feat` 的布局为 `[3B,768]`。按原训练循环的真实输出顺序还原为 `f_R, f_N, f_T`，不能假定数据加载器返回顺序与本地代码无关。

### 3.2 元数据

新损失至少需要每张图像的：

- 连续训练 PID；
- modality ID；
- camera ID（必须使用与正式 evaluator 相同的 camera ID 语义/映射；若原始编号是模态局部编号，先转换为 evaluator 使用的统一编号）；
- 稳定 sample ID 或路径索引（用于 Mixed 自匹配屏蔽）。

所有元数据必须与拼接后的图像顺序一致。原始 PID 不直接作为 Bank 索引，除非数据集已保证其连续且与 Classifier 类别顺序一致。

### 3.3 L2 特征

$$
u_i=\operatorname{L2}(f_i)
=\frac{f_i}{\max(\|f_i\|_2,\epsilon)}.
$$

MP 与 Pair/Mixed 使用 `u`。WRT 与 ProCA/GPD 继续按原 UAD 的输入和内部归一化逻辑运行。不能为了代码复用而把所有损失强行改成同一特征版本。

---

## 4. 原 UAD 损失的保留要求

总目标：

$$
\mathcal L_A =
\mathcal L_{CE}+\mathcal L_{WRT}
+0.01\mathcal L_{ProCA}+\mathcal L_{GPD}
+\lambda_P(t)\mathcal L_{MP}
+\lambda_C(t)\mathcal L_{img}.
$$

### 4.1 CE

继续用原 `cls_score` 计算身份交叉熵。梯度经过 Classifier、BNNeck、pre-BN CLS 和共享 ViT。

### 4.2 WRT

直接调用原 WRT 实现。保持原距离计算、正负样本权重和输入特征；不要把它替换成 Pair/Mixed，也不要改变 WRT 的权重。

### 4.3 ProCA

保留原中心构造和 detach 逻辑：

1. 每张图像特征先 L2；
2. 按 PID 与模态聚合 K 张图像；
3. 对模态中心再次 L2；
4. 通过三个模态中心构造统一身份中心并 detach；
5. 计算原 ProCA 损失；
6. 原权重 0.01。

### 4.4 GPD

保留原 GPD 温度 0.03、原型 Bank、中心聚合与 EMA 公式。原配置 `GPD_MOMENTUM=0.2` 表示新中心系数 0.2、旧 Bank 系数 0.8。GPD Bank 按原 UAD 方式初始化；不要用 MP Bank 或 ImageNet bootstrap pass 替换它。B0/A1 共同增加成功 optimizer step 门控：只有参数更新成功时才执行原 GPD Bank 更新，AMP 跳步时不更新。

---

## 5. MP Bank 与 MP 损失

### 5.1 数据结构

- `MP_Bank`: `[3, num_train_ids, 768]`，FP32、单位向量；
- `MP_valid`: `[3, num_train_ids]`，bool；
- `MP_momentum_new=0.2`；
- `tau_P=0.1`；
- `lambda_P=0.1`；
- `lambda_TT=0.1`。

Bank 是无梯度状态，不注册成可学习 Parameter。若使用 buffer，需明确它的保存、恢复与 DDP 同步方式。

### 5.2 EMA

对每个有效 `(PID, modality)` 中心：

$$
P_{y,m}\leftarrow
\operatorname{L2}(0.8P_{y,m}+0.2c_{y,m}).
$$

第一次出现且该行 invalid 时，直接用 `L2(c)` 初始化并设置 valid=True。当前 iteration 中来自多个 rank 的同一 `(PID, modality)` 中心必须先聚合，再更新一次；不得按 rank 顺序重复 EMA。

### 5.3 目标模态 softmax

对方向 $m\to n$：

$$
s_{i,z}=u_i^m\cdot P^M[n,z]/\tau_P.
$$

只保留 `MP_valid[n,z]=True` 的类别。正类原型无效的 Query 跳过该方向，不得让正类缺席但继续计算 CE。目标原型全部为单位向量；计算 logits 与交叉熵时使用 FP32。

六方向等权平均；TIR→TIR 作为轻量辅助：

$$
\mathcal L_{MP}
=\operatorname{Mean}_{m\ne n}\mathcal L^P_{m\to n}
+0.1\mathcal L^P_{T\to T}.
$$

若一个方向没有有效 Query，则跳过该方向。若所有方向都无有效 Query，返回与当前特征计算图安全连接的零损失，并记录告警；不能产生 NaN。

---

## 6. Pair/Mixed 图像级对比实现

### 6.1 相似度

$$
S^{m,n}=U_mU_n^\top/\tau_C,
\qquad \tau_C=0.1.
$$

`U_m`、`U_n` 是 `[B,768]` L2 pre-BN CLS 特征。相似度矩阵、mask、`logsumexp` 在 FP32 计算。

### 6.2 有效候选

与实际 evaluator 对齐：

$$
V_{ij}=\mathbf1[c_i\ne c_j].
$$

有效正样本：

$$
P_{ij}=V_{ij}\mathbf1[y_i=y_j].
$$

有效负样本：

$$
N_{ij}=V_{ij}\mathbf1[y_i\ne y_j].
$$

Mixed 中额外屏蔽同一图像作为 Query 与 Gallery 的自匹配。相机掩码不能代替自匹配掩码。

只有至少有一个有效正样本和一个有效负样本的 Query 才计算损失。正样本、负样本的有效性都要体现在 softmax 分母候选集合中；不能仅屏蔽正样本。无效 Query 在 masked softmax 前排除。

### 6.3 Uniform/Set

$$
\mathcal L_{uni}^{(i)}
=-\frac1{|\mathcal P_i|}
\sum_{j\in\mathcal P_i}
\log\frac{\exp(S_{ij})}{\sum_{k\in\mathcal V_i}\exp(S_{ik})}.
$$

$$
\mathcal L_{set}^{(i)}
=-\log\frac{\sum_{j\in\mathcal P_i}\exp(S_{ij})}
{\sum_{k\in\mathcal V_i}\exp(S_{ik})}.
$$

$$
\mathcal L_{img}^{(i)}
=0.3\mathcal L_{uni}^{(i)}+0.7\mathcal L_{set}^{(i)}.
$$

### 6.4 Pair

依次计算六个方向：

- RGB→NIR；
- NIR→RGB；
- RGB→TIR；
- TIR→RGB；
- NIR→TIR；
- TIR→NIR。

每个方向先对有效 Query 求均值，再对有效方向求均值：

$$
\mathcal L_{pair}=\operatorname{Mean}_{m\ne n}\mathcal L_{img}(U_m,U_n).
$$

### 6.5 Mixed

$$
U_{mix}=[U_R;U_N;U_T].
$$

分别计算 RGB→Mixed、NIR→Mixed、TIR→Mixed，并对三个源模态等权平均：

$$
\mathcal L_{mixed}
=\operatorname{Mean}_{m\in\{R,N,T\}}
\mathcal L_{img}(U_m,U_{mix}).
$$

最终：

$$
\mathcal L_{img}=0.75\mathcal L_{pair}+0.25\mathcal L_{mixed}.
$$

### 6.6 梯度检查

Pair/Mixed 必须让两端特征都保持梯度。不得对 Gallery 特征整体 `detach()`。只有标签、掩码和原型 Bank 无梯度。单元测试中应分别确认 Query 端与 Gallery 端的梯度非零。

---

## 7. PKM 兼容的相机多样性 K-group 与软困难 PID 调度

### 7.1 设计约束：不再在线重采样 PID

不能在每个 batch 直接从当前可用 PID 池重新抽取“8 个随机 + 8 个困难”并声称只改变共现关系。这样可能让困难 PID 提前耗尽 K-group，导致剩余 group 无法组成完整 batch，改变 epoch 长度、PID 曝光频率和样本消耗。

A1 必须先生成原 UAD PKM 在该 rank、本 epoch 实际会消耗的基准 schedule，再只交换 schedule 中已有的 PID-group 记录。这样才能保留原 PKM 的 epoch 账本。

### 7.2 基准 schedule 物化与样本账本

1. 调用/复用原 UAD PKM 的 group pool 构造、PID 可用性判断、每 batch PID 数、三模态顺序、每 PID 每模态 K 张图像、group 耗尽与尾部处理逻辑，物化完整 epoch schedule。
2. 每条 PID-group 记录包含一个 PID 与该 PID 在 RGB/NIR/TIR 三个模态各 K 张图像的索引。只保留原 PKM schedule 本来会消耗的记录；原始规则会留下未使用 group 时，不额外补齐。
3. 物化后记录：总 batch 数、每个 PID 的出现次数、每 PID 的模态 group 数、每个样本索引的 occurrence 次数、每个 rank 的 iteration 数。后续操作均不得改变这些统计。原 PID schedule 使用与 B0 相同的 sampler seed；相机重分组和困难交换各使用独立 RNG 流，不得消耗或改变基准 sampler 的随机状态。
4. 如果现有 sampler 是惰性 generator，可在 epoch 开始时将其输出物化为 list，再进行受约束的重组；不得自行另写一个不同的 epoch 终止条件。

### 7.3 相机多样性优先的 K-group 重分组

A1 从 Epoch 1 起使用此步骤，目的是提高同一 PID/模态的 K 张图像来自不同相机的机会，从而增加 Pair/Mixed 获得有效跨相机正样本的概率。

1. 对每个 PID、每个模态，收集基准 schedule 中本 epoch 实际会消耗的所有 K-group 图像 occurrence。occurrence 是样本索引的一次出现；如果原 PKM 因样本不足而重复某个索引，重复 occurrence 必须保留。
2. 将该 PID/模态所有已消耗 group 的 occurrence 展平，再重新分配到相同数量、每组恰好 K 个 occurrence 的新 group。总 occurrence 数、每个样本索引的出现次数和 group 数必须完全一致。
3. 逐组贪心构造：每次优先从尚未在当前 group 出现的 camera ID 中随机选一个 occurrence；若没有新的 camera ID 可选，再从所有剩余 occurrence 中随机补足。随机并列选择由 `seed + epoch + rank + PID + modality + 'camera_repack'` 派生的独立 RNG 控制；该 RNG 与原 PKM schedule 生成 RNG 分离。
4. 相机数量不足时允许同一 group 中出现重复 camera；不人工复制图像、不丢弃样本，也不强制不同平台配对。
5. 将重分组后的 K-group 依次回填到该 PID 在原 schedule 的 group 槽位。三模态仍按原 UAD 的 RGB/NIR/TIR 顺序对应。该操作不改变 ProCA 公式，但会改变 K 张图像如何共同形成某次 ProCA 模态中心，这是 A1 明确采用的采样适配。
6. 完成后断言：每 PID/模态 group 数、每 PID 出现次数、每样本 occurrence 次数、batch 数、每 batch PID 数与基准账本一致。

相机多样性只提高有效正样本出现机会，不保证每个 Query 都有有效跨相机正样本。损失仍按第 6 节掩码规则过滤无效 Query，并记录有效率。

### 7.4 Epoch 1–10：随机 PID schedule，不启用困难交换

Epoch 1–10 保留原 PKM 的随机 PID batch schedule，不进行困难共现交换；但 A1 的 K-group 已按第 7.3 节做相机多样性重分组。B0 从始至终使用原始 PKM 的 PID 调度与原始 K-group 构造。

### 7.5 Epoch 10 后的固定困难参考：DDP 安全生成

Epoch 10 结束后，各 rank 完成训练与 Bank 更新并同步。先保存 Epoch 10 结束时的模型 state 和哈希；随后 rank 0 使用当前模型的未包装实例 `model.module` 对完整训练划分做确定性特征提取：

- 独立确定性数据视图、顺序遍历、`drop_last=False`，不使用 `DistributedSampler`，不只提取 rank 0 分片；
- `eval()` + `torch.inference_mode()`，不调用 DDP wrapper forward；
- 只使用训练图像与训练 PID；不使用测试 Query/Gallery；
- 使用独立 DataLoader generator，必要时保存/恢复 RNG 状态；
- 从 pre-BN CLS 构造全局身份中心和 RGB/NIR/TIR 模态身份中心。

困难度定义：

$$
h_G(y,z)=\cos(\bar p_y^G,\bar p_z^G),
$$

$$
h(y,z)=0.5h_G(y,z)+0.5h_T(y,z),
$$

其中 `h_T` 是有效 TIR↔TIR 及涉及 TIR 的跨模态身份中心相似度平均，构造时同时纳入身份对的两个顺序，并最终显式对称化 `h=(h+h^T)/2`。无有效项时退回有效全局困难度；保存前必须确保矩阵没有 NaN/Inf。

保存 `hard_reference_epoch010.pt`，至少包含中心、困难度矩阵、训练 split/manifest 哈希、PID 映射、模态/相机元数据映射、预处理配置、提取特征所用 Epoch 10 模型 state 哈希、生成 epoch、`τ_s`、构建代码版本与文件 SHA-256。rank 0 保存后，将路径/哈希和矩阵广播给所有 rank，或让各 rank 从共享文件系统读取并核验哈希；随后保存带有该 artifact 路径/哈希的完整 Epoch 10 训练 checkpoint，barrier 后才进入 Epoch 11。

固定参考在本次完整训练内不刷新。断点恢复必须加载原 artifact 并核验哈希，不能悄悄用恢复时的当前模型重新计算。如果 artifact 丢失，只能从保存的 Epoch 10 模型状态按同一确定性流程显式重建并记录；无法确认模型状态一致时 fail-fast。checkpoint 需保存 artifact 路径与哈希。

### 7.6 Epoch 11–120：受约束的软困难共现交换

定义 batch 的平均困难度：

$$
H(B)=\operatorname{Mean}_{y<z,\,y,z\in B}h(y,z).
$$

执行流程：

1. 每 epoch 将约 50% 的 PID-group 记录位置设为困难化候选位置并打乱顺序。`hardening_slot_fraction=0.5` 是尝试困难化的位置比例，不是强制每个 batch 恰好 8 个 PID 由困难抽样产生。
2. 对候选记录 `y`，从其他 batch 中最多随机抽取 32 条 group 记录作为候选 `z`。只考虑交换后两个 batch 都不出现重复 PID 的候选。
3. 计算交换前后两个 batch 的困难度变化：

$$
\Delta=H(B_a^{swap})+H(B_b^{swap})-H(B_a)-H(B_b).
$$

4. 仅从 `Δ>0` 的可行交换中按 `softmax(Δ/τ_s)` 采样，`τ_s=0.2`。若没有正增益交换，则保留原记录。交换抽样使用独立的 `seed + epoch + rank + 'hard_swap'` RNG；不得改变后续 epoch 的原 PKM 随机状态。每条 group 记录在本轮困难化阶段最多参与一次交换。
5. 交换的是完整 PID-group 记录，不是新抽取图像；因此 PID 每 epoch 出现次数、每个样本 occurrence 次数、每 PID 各模态 group 数、每 batch PID 数、batch 总数和每 rank iteration 数完全不变。每次交换后运行账本断言。
6. 所有 rank 在 Epoch 10 边界加载同一困难参考，但只在各自原有 rank schedule 内交换，不跨 rank 移动数据。这样不会改变原 DDP 的数据分片与 collective 次数。

该方案的目标是增加难身份共现，不是保证所有 batch 都达到最大困难度。若可行交换不足，保留原随机 schedule 优先于破坏 PKM 约束。

## 8. 训练循环接线、AMP 成功步检测与 Bank 更新顺序

每个 iteration 按以下顺序执行：

1. 原 UAD/A1 sampler 产生 RGB/NIR/TIR 图像与 PID、camera、sample ID；A1 的 K-group 已按相机多样性策略构造，Epoch 11 起 schedule 还经过受约束 group 交换。
2. 原 UAD forward 得到 `cls_score`、`global_feat`、`feat`。
3. 按原拼接顺序拆分 `global_feat`，计算 `U_R/U_N/U_T=L2(global_feat)`。
4. 计算原 CE、WRT、ProCA；用更新前的 GPD Bank 计算 GPD；用更新前的 MP Bank 计算 MP；再计算 Pair 与 Mixed。
5. 合并：

$$
\mathcal L_A=\mathcal L_{CE}+\mathcal L_{WRT}+0.01\mathcal L_{ProCA}+\mathcal L_{GPD}+\lambda_P(t)\mathcal L_{MP}+\lambda_C(t)\mathcal L_{img}.
$$

6. AMP backward；若原配置使用梯度裁剪，先 `scaler.unscale_(optimizer)` 再按原规则裁剪。不要因为新增损失额外引入不同的裁剪配置。
7. **检测优化步骤是否成功：**
   - AMP 开启：在 `scaler.step(optimizer)` 前记录 `old_scale=scaler.get_scale()`；调用 `scaler.step(optimizer)`，再调用 `scaler.update()`，读取 `new_scale=scaler.get_scale()`。
   - `new_scale < old_scale` 表示本步因非有限梯度被 GradScaler 跳过；`new_scale >= old_scale` 表示本步成功。不能使用 `scaler.step()` 返回值判断，因为普通 optimizer 的 `step()` 通常返回 `None`。
   - AMP 关闭：若 `optimizer.step()` 正常执行且未抛出异常，则视为成功。
   - DDP 下对成功标志做 all-reduce MIN/MAX 一致性检查。若 rank 之间判断不一致，所有 rank 跳过 Bank 更新并 fail-fast，防止记忆状态分叉。
8. 只有在优化步骤成功时才执行 Bank 更新：先汇总各 rank 当前 batch 的 detached `(PID, modality, center)`，合并重复 PID；再按原 GPD 公式/EMA 更新 GPD Bank，按 A-V1 EMA 更新 MP Bank。一个 `(PID, modality)` 在一次成功 iteration 中最多更新一次。
9. 若优化步骤被跳过，GPD 与 MP 两类 Bank 都不更新；记录 AMP skip 计数。B0 也使用同一成功步门控更新 GPD Bank，以保证 B0/A1 比较条件一致。GPD 的原型定义、中心聚合、EMA 系数与更新公式不变；成功步门控是两组共同采用的数值稳定性处理。
10. 记录各项 loss、各方向有效 Query 数/比例、Bank 有效覆盖、PID-group 账本、困难交换次数、AMP 成功/跳过次数和 Bank 更新次数。

### 8.1 Checkpoint 状态要求

为了支持断点后沿着同一训练轨迹继续，B0/A1 的训练状态 checkpoint 应至少保存：模型、optimizer、scheduler、GradScaler、epoch、随机数状态、GPD Bank 及其有效性状态（若原实现有）、A1 的 MP Bank/valid、当前 sampler 阶段、每 epoch 的确定性 seed、困难参考 artifact 路径与哈希。若在 epoch 中间保存，还必须保存 sampler 已消耗位置/剩余 group 状态；优先在 epoch 边界保存，以避免额外的中途 sampler 状态管理。

恢复时：

- 不重新执行 ImageNet bootstrap 覆盖已训练的 MP Bank；
- Epoch 11 及之后必须加载原固定困难参考 artifact 并核验哈希；
- 恢复 optimizer、scheduler、GradScaler 与 RNG 状态；
- 恢复后核对下一 epoch 的 batch 数、PID 出现次数和 rank 步数；
- 若关键 Bank 或困难参考丢失且无法从完全一致的 checkpoint 恢复，明确报错，不静默重新初始化。

## 9. 学习率与完整训练配置

A1 沿用当前原 UAD 完整训练命令与配置：

- 从 ImageNet 预训练权重初始化；
- 完整 120 Epoch；
- 原 UAD SGD、参数组、weight decay、bias LR factor、warm-up 和 scheduler；
- 原 UAD batch 组织与图像增强；
- 原 ProCA/GPD 配置；
- 训练输出与 checkpoint 周期尽量沿用原配置。

新增损失的权重单独 warm-up：

$$
r(t)=\min(t/5,1),\quad
\lambda_P(t)=0.1r(t),\quad
\lambda_C(t)=0.1r(t).
$$

该 warm-up 只作用于新增损失，不改变原学习率 scheduler。训练 Epoch 从 1 开始计数：第 1 Epoch 的新增权重为 0.02，第 5 Epoch 起达到 0.1。

---

## 10. B0/A1 对照与评测

### 10.1 B0

- ImageNet 初始化；
- 原 UAD 完整训练；
- 原随机 PKM；
- 原 CE/WRT/ProCA/GPD；
- 原 GPD Bank 公式/EMA 行为，并与 A1 使用相同的 AMP 成功步更新门控；
- 原优化器、scheduler、训练轮数与增强；
- 原评测协议。

### 10.2 A1

- 同一 ImageNet 初始化；
- 同一完整训练配置与预算；
- 保留所有原 UAD 损失；
- 新增 MP、Pair/Mixed；
- A1 从 Epoch 1 起做相机多样性 K-group 重分组；Epoch 1–10 保持原随机 PID schedule，Epoch 11 起进行受约束软困难 group 交换；
- GPD Bank 初始化/公式不变，且与 B0 共用 AMP 成功步门控；另有 MP Bank；
- 相同评测协议。

不能将 A1 的完整训练结果与 B0 的短训练、不同 batch/LR 或不同评测协议直接比较。

### 10.3 评测

- `NECK_FEAT=before`；
- `FEAT_NORM=yes`；
- 768D pre-BN CLS；
- L2 + 欧氏距离；
- 官方/本地正式 evaluator 同相机过滤；
- 无 rerank、无 TTA；
- 固定 Query/Gallery；
- 输出总体 mAP、Rank-1、Rank-5、Rank-10、mINP；
- 输出九种有向模态组合的 Rank-1/mAP/mINP。

---

## 11. 单元测试与 Smoke Test

正式训练前至少检查：

### 11.1 特征与标签

- [ ] RGB/NIR/TIR 的 `global_feat` 拆分顺序正确；
- [ ] PID、modality、camera、sample ID 与图像一致；
- [ ] 连续 PID 映射与 Classifier、Bank 索引一致；
- [ ] 每个 modality 的特征形状为 `[B,768]`；
- [ ] 所有 L2 特征范数有限且接近 1。

### 11.2 MP

- [ ] bootstrap pass 只读取训练划分，不执行 backward；
- [ ] 缺失模态的原型保持 invalid；
- [ ] invalid 原型不进入 softmax 分母；
- [ ] 六方向与 TIR→TIR 的标签索引正确；
- [ ] MP 梯度回到 Query 特征；
- [ ] MP Bank 无梯度；
- [ ] EMA 每个有效 `(PID, modality)` 每个成功 iteration 最多更新一次。

### 11.3 Pair/Mixed

- [ ] 同相机 Gallery 候选被屏蔽；
- [ ] Mixed 自匹配被额外屏蔽；
- [ ] 无正样本或无负样本的 Query 被跳过；
- [ ] invalid 候选不进入 softmax 分母；
- [ ] Uniform/Set 比例为 0.3/0.7；
- [ ] Pair/Mixed 比例为 0.75/0.25；
- [ ] Query 与 Gallery 两端梯度都能回传；
- [ ] 没有 NaN/Inf。

### 11.4 采样器与 DDP

- [ ] 物化的基准 schedule 与原 UAD PKM 的 batch 数、PID 出现次数和 `(PID, modality, sample_index)` occurrence 账本一致；
- [ ] A1 的相机多样性重分组不增删 occurrence，K-group 数和每组 K 不变；
- [ ] 有多相机候选时优先覆盖不同 camera，camera 不足时允许重复；
- [ ] Epoch 1–10 不做困难交换，Epoch 11 起才启用固定参考驱动的 group 交换；
- [ ] 每次交换后 batch 内 PID 唯一，且 PID 频率、group 账本、batch 数与 rank 步数不变；
- [ ] 所有 rank 的困难参考路径/哈希一致；
- [ ] Epoch 10 参考由 rank 0 对完整训练划分提取，未调用 DDP wrapper forward，也未只提取本地分片；
- [ ] 断点恢复加载原困难参考，不静默重新计算。

### 11.5 完整训练与 AMP/Bank 状态

- [ ] A1/B0 均从 ImageNet 权重开始；完整训练 Epoch、batch、LR、scheduler 与增强一致；
- [ ] 没有意外加载 ReID checkpoint；
- [ ] B0/A1 的 GPD 公式与 EMA 一致，且都仅在成功 optimizer step 后更新；
- [ ] A1 的 MP Bank、valid mask 与固定困难参考元数据按要求保存；
- [ ] AMP 跳步时 GPD/MP Bank 均不更新；DDP 各 rank 对成功标志一致；
- [ ] 恢复训练时 epoch、optimizer、scheduler、GradScaler、RNG、Bank、困难参考和 sampler 阶段一致。

### 11.6 新增损失梯度贡献检查

损失数值下降并不保证新增机制对 CLS 产生了有效梯度。只在 Smoke Test 和训练早期少量 checkpoint 上检查，不在每个 iteration 额外计算梯度。

1. Smoke Test 选择一个确实有有效 Query 的 batch；另保存 Epoch 1 与 Epoch 5 的模型状态、对应确定性 batch、当时的 GPD/MP Bank 与 valid mask 快照、PID/camera/sample ID 元数据和新增损失权重，用单进程诊断副本复测。这样避免在主 DDP 训练循环中插入多次反向传播或 collective。
2. 对同一 `global_feat` 分别计算以下加权损失的梯度范数：
   - `L_base = CE + WRT + 0.01 * ProCA + GPD`；
   - `lambda_P * MP`；
   - `lambda_C * 0.75 * L_pair`；
   - `lambda_C * 0.25 * L_mixed`。
3. 可使用 `torch.autograd.grad(loss_component, global_feat, retain_graph=True, allow_unused=True)` 获取梯度。该诊断只读取返回的中间特征梯度，不调用 optimizer，不更新 Bank，不写入参数 `.grad`；随后在正式训练流程中仍只对总损失做一次正常 backward。
4. 记录原始 loss、加权 loss、梯度范数、有限性、相对于 `L_base` 的梯度范数比，以及 MP 六方向、Pair 六方向和 Mixed 三方向的有效 Query 数/比例。
5. MP/Pair/Mixed 梯度应有限；在有有效 Query 且损失确实依赖 `global_feat` 时应非零。若为零，先查有效样本、相机 mask、MP valid、特征是否 detach 或损失是否未接入总损失。
6. 若新增项相对基础损失的梯度贡献在多个有效 batch 中接近数值零，或与基础梯度相差多个数量级，先核查接线与有效样本，而不是立即调权重。梯度范数是局部诊断指标，不是必须达到某个固定比例的优化目标，也不能替代正式检索指标。

## 12. 结果记录与决策规则

每次运行记录：

- 数据集、训练划分、模型起点权重哈希；
- 配置文件、启动命令、随机种子；
- Epoch、每 Epoch iteration、全局 batch；
- 原始四项损失与新增 MP/Pair/Mixed 分项；
- 六方向 MP 有效 Query 数；
- Pair/Mixed 各方向有效 Query 数；
- MP Bank 有效身份覆盖率；
- MP bootstrap artifact 路径/哈希、训练 split/manifest 哈希与预处理配置；
- Epoch 10 困难参考 artifact 路径/哈希、模型 checkpoint 哈希与生成配置；
- 原 PKM 账本与 A1 重组后的 batch 数、每 PID 出现次数、每样本 occurrence 次数；
- 相机多样性前后每 PID/模态 K-group 的相机数统计；
- 困难交换候选数、成功交换数、平均 Δ 与平均 batch 困难度；
- 六方向 Pair、三个 Mixed 方向的有效 Query 数/比例；
- AMP skip 次数、成功 optimizer step 次数、GPD/MP Bank 更新次数；
- Smoke/Epoch 1/Epoch 5 的加权梯度范数与相对基础梯度比；
- 总体五项指标、九组合指标；
- 最终 checkpoint 与训练日志路径。

| 观察结果 | 处理 |
|---|---|
| 总体指标与主要困难方向协同提升 | 扩展到 1000-GD 与 2337，保持配置不变 |
| mAP 上升但 Rank-1 下降 | 优先检查 Uniform/Set 与困难负类分布，不立刻增加新模块 |
| Rank-1 上升但 mINP 下降 | 检查长尾正样本、跨相机有效正样本与 Mixed 候选 |
| 训练损失下降而检索不变 | 检查新监督是否提供了原 UAD 之外的有效增量 |
| 明显退化 | 先检查初始化、数据/标签顺序、mask、Bank 更新、学习率配置和梯度量级 |
| 九组合收益不均 | 判断是否由某些方向退化抵消总体收益，再决定是否需要后续针对性研究 |

---

## 13. 论文与代码依据

### UAD / WHU-MARS

- 论文：<https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_WHU-MARS_A_Multispectral_Aerial-Ground_Benchmark_Towards_Any-Scenario_Person_Re-Identification_CVPR_2026_paper.pdf>
- 官方仓库：<https://github.com/msm8976/WHU-MARS>
- UAD 配置：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/configs/UAD.yml>
- 训练说明：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/README.md>
- ProCA/GPD/训练循环：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/processor/processor.py>
- 模型结构：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/model/make_model.py>
- PKM/DDP：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/datasets/sampler_ddp.py>
- WRT：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/loss/wrt_loss.py>
- 评测：<https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/utils/metrics.py>

### MINER

- 官方仓库：<https://github.com/oronnir/MINER>
- 软困难采样：<https://github.com/oronnir/MINER/blob/main/miner_tvrid_rgb_final/sampler.py>
- 多正样本对比：<https://github.com/oronnir/MINER/blob/main/miner_tvrid_rgb_final/losses.py>
- 训练实现：<https://github.com/oronnir/MINER/blob/main/miner_tvrid_rgb_final/train.py>

**实现边界：** UAD 的模型、原损失定义、ProCA/GPD 公式与 EMA、评测定义保留。为训练状态一致性，B0/A1 共同增加 AMP 成功步门控：只有 optimizer 成功更新后才更新 GPD Bank；这是共享的数值稳定性处理，不是新增研究损失。MINER 的多正样本对比与软困难身份选择只作为机制参考。MP Bank、六方向 MP、TIR→TIR、Pair/Mixed、Uniform/Set 混合、WHU 相机掩码、相机多样性 K-group 重分组、受约束跨 batch group 交换以及 Epoch 10 后固定困难参考，均为 A-V1 的任务适配实现，不是原论文代码中已经存在的模块。
