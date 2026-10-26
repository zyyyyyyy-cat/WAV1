# A-V1 代码修改记录

## 1. 实现依据与取舍

优先按 `idea.md` 和 `guide.md` 实现，在 UAD 的单分支 ViT-B/16、CE/WRT/ProCA/GPD、BNNeck 和原评测器上增加 MP、Pair/Mixed、相机多样性 K-group 与软困难 PID 共现。WHU-MARS 论文 Table 8 中，ProCA 和 GPD 合用优于单独添加，因而未替换原有中心与全局 Bank。MINER 的多正样本对比与软困难抽样有独立消融支持；本实现保留其随机困难选择思想，但遵守 WHU 的三模态 PKM 账本与同相机过滤规则。MINER 的服装采样、双编码器和方差控制器没有搬入。

注意：当前本地 WHU 训练集模态名是 `RGB/IR/Thermal`，新增模块内部索引 0/1/2 分别对应论文叙述的 RGB/NIR/TIR；相机 ID 沿用数据集文件名中的全局编号减一后的值，与正式 evaluator 一致。

## 2. 文件清单

| 文件 | 变更及目的 |
| --- | --- |
| `WHU/loss/av1.py`（新增） | MP 原型 Bank、七方向 MP（六个跨模态方向与轻量 TIR→TIR）、六方向 Pair、三方向 Mixed；FP32 logits，过滤无效原型、同相机与自匹配，跳过无有效正负样本的 Query。 |
| `WHU/processor/av1_state.py`（新增） | 确定性完整训练集特征提取、manifest/文件/模型哈希、RNG 保存恢复、Epoch 10 固定困难矩阵。 |
| `WHU/datasets/av1_sampler.py`（新增） | 先物化原 PKM 实际完整 batch，再保持 PID-group 与样本 occurrence 账本做相机 K-group 重组；Epoch 11 起执行最多 32 候选、软最大正增益交换。 |
| `WHU/datasets/bases.py` | 按训练 manifest 为每个模态预计算不重叠的 ID 区间，训练样本为每张原始图像提供稳定 sample ID；验证 PKM 元组内 PID 一致。 |
| `WHU/datasets/make_dataloader.py` | collate sample ID；A1 时包装原单卡或 DDP PKM sampler；提供完整训练划分的确定性三模态 loader，仅用于离线原型与困难参考。 |
| `WHU/processor/processor.py` | 原损失后接入新损失、五 epoch 权重 warm-up、AMP 成功步 GPD/MP 更新门控、DDP 一致性检查、bootstrap、Epoch 10 参考生成、训练状态保存/恢复与覆盖日志。 |
| `WHU/config/defaults.py` | 新增默认关闭的 `AV1` 配置，不改变 `UAD.yml` 的默认运行模式。 |
| `WHU/configs/A1.yml`（新增） | 与 `UAD.yml` 相同的基础训练配置，启用 A1，设置新增损失温度与权重，输出到独立目录。 |
| `WHU/train.py` | A1 起点和模态顺序校验；传入完整训练划分 loader。 |
| `WHU/tests/test_av1.py`（新增） | 合成特征梯度、无效相机候选、Bank 有效性、采样出现次数，以及真实 `ImageDataset → collate → Mixed` 的单图像 ID 与掩码测试。 |

没有删除原 WHU 文件，也没有改动 `WHU/test.py`、模型结构、WRT 实现或评测器。

## 3. 模型、前向与反向路径

输入按 `[RGB; IR; Thermal]` 拼接。原模型仍返回分类 logits、最终 LayerNorm 后且 BNNeck 前的 768 维 CLS、BNNeck 后特征。CE 走 classifier/BNNeck；WRT、ProCA、GPD 走原 `global_feat` 路径；MP 和 Pair/Mixed 对同一 `global_feat` 逐图 L2 后计算。推理仍用原配置的 768 维 pre-BN CLS、L2 与欧氏距离，不加载新增 Bank 或相机标签。

总损失为

`L = L_CE + L_WRT + 0.01 L_ProCA + L_GPD + 0.1 min(epoch/5,1) L_MP + 0.1 min(epoch/5,1) (0.75 L_pair + 0.25 L_mixed)`。

MP 在目标模态有效原型上做 `softmax(u_query · P_target / 0.1)`，六个跨模态方向等权平均，另加 `0.1 L_TIR→TIR`。Bank 为 `[3,C,768]` FP32 buffer 和 `[3,C]` 有效掩码，不接收梯度。其初始化只读完整训练集的确定性 pre-BN CLS，先逐图 L2、按 PID/模态求均值、再 L2。每次成功优化步后，跨 rank 合并同一 PID/模态中心并只做一次 `L2(0.8 P_old + 0.2 P_new)`；首次有效行直接赋值。

Pair/Mixed 使用 `S = u_q u_g^T / 0.1`。候选必须相机不同；Mixed 另屏蔽相同 sample ID。每个有效 Query 必须同时有跨相机正样本与负样本。Uniform 项是所有正样本的平均负 log-softmax，Set 项是 `logsumexp(所有有效候选) - logsumexp(有效正样本)`，两项比例为 0.3/0.7。Query 与 Gallery 特征都不 detach；梯度能从损失回到共享 CLS 和 ViT。

### 3.1 单张图像 ID 与 Mixed 自匹配

`ImageDataset` 接收的元组是三个各自模态列表中的**原始图像索引**，而非一个图像 ID。初始化时，为 `RGB/IR/Thermal` 分别分配长度等于对应训练列表长度的不相交整数区间。取样时生成 `sample_id_m = offset_m + index[m]`。因此同一 PKM 元组的三张不同模态图像 ID 不同；同一原始图像无论在哪个 epoch、哪个 rank、哪个 PKM 元组中被重复抽样，ID 都相同。collate 将三个模态的 ID 各整理成 `[B]`，顺序与 `imgs`、`camids`、`global_feat` 的 `[RGB; IR; Thermal]` 拼接顺序一致。

Mixed 中候选掩码是 `camera_q != camera_g AND sample_id_q != sample_id_g`。例如训练列表各有 3 张时，同一个 PKM 元组 `(0,0,0)` 返回的三张图像 ID 为 `(0,3,6)`：RGB Query 0 对其自身 Gallery 0 被排除；如相机不同，对该元组 IR Gallery 3 和 Thermal Gallery 6 均保留。这里的 ID 只属于图像，不属于三模态组合或 batch 位置。

### 3.2 完整正向数据流

1. 原 PKM 采样器产生每身份、每模态 K 张图像的索引；A1 先物化原实际消耗的完整 batch，对每 PID/模态已有 occurrence 做相机重组，Epoch 11 起可在同一 rank 的 batch 之间交换完整 PID-group。B0 使用原 PKM。
2. `ImageDataset` 对每个三模态索引元组读取三张图像，返回三张增强后的图像、同一连续训练 PID、三个相机 ID、三个单图像 ID。collate 得到三组 `[B,3,H,W]`、`target[B]`、三个 `camids[B]` 和三个 `sample_ids[B]`。
3. 原模型按 `[RGB; IR; Thermal]` 拼成 `[3B,3,H,W]`，共享 ViT-B/16 输出最终 LayerNorm 后的 `global_feat[3B,768]`；BNNeck 给出 `feat[3B,768]`，Classifier 给出 `cls_score[3B,C]`。目标 PID 同序复制三份。
4. CE 用 `cls_score`；WRT 用原 `global_feat` 距离；ProCA 将每张 `global_feat` 先 L2，再按模态/PID/K 求中心并再次 L2，三模态均值身份中心 detach；GPD 对原 `global_feat` 逐图 L2，用更新前的全局身份 Bank 算温度为 0.03 的分类损失。上述路径与原 UAD 相同。
5. A1 对 `global_feat.float()` 逐图 L2，得到 `U_RGB/U_IR/U_Thermal`。MP 用更新前的目标模态原型 Bank 计算六个跨模态方向和 TIR→TIR；无效原型不进 softmax 分母，正类原型无效的 Query 不参与。
6. Pair 对六个有向模态对、Mixed 对三个源模态到 `[U_RGB;U_IR;U_Thermal]` 计算相似度。候选按相机与单图像 ID 过滤，再以 PID 划分正负；缺正样本或缺负样本的 Query 跳过。有效 Query 的 Uniform/Set 项按 0.3/0.7 混合，Pair/Mixed 按 0.75/0.25 混合。
7. 原四项损失与经五 epoch warm-up 的 MP、Pair/Mixed 相加，得到唯一总损失；推理仍走原模型和原评测器，不使用新增 Bank、相机或 sample ID。

### 3.3 完整反向与状态更新流

1. 总损失执行一次 AMP 缩放后的 `backward()`。CE 的梯度沿 Classifier → BNNeck → `global_feat` → ViT；WRT、ProCA、GPD、MP、Pair/Mixed 都沿各自对 `global_feat` 的计算回到共享 ViT。ProCA 的统一身份目标中心已 detach，只有模态中心及其图像特征接收该项梯度。GPD 的 Bank 和 MP 的 Bank/有效掩码是无梯度参考，只有当前 Query 特征接收原型分类损失的梯度。
2. Pair/Mixed 的 Query 与 Gallery 都来自同一次模型前向，均不 detach；一个图像可同时以 Query 和 Gallery 身份收到多个方向的梯度，autograd 将这些贡献累加到对应 `global_feat` 行。相机/PID/sample ID 比较产生布尔掩码，不接收梯度；被屏蔽候选不进 softmax 分母，跳过的 Query 不产生该方向梯度。
3. AMP unscale 后按原 UAD 规则对模型参数裁剪梯度，再由 optimizer 更新模型参数；DDP 在反向期间同步参数梯度。`GradScaler` scale 下降表示本步跳过，跨 rank 对成功标志做一致性检查。
4. 只有成功优化步才以本步前向得到的 detached 特征更新 GPD 与 MP Bank。GPD 保留原中心聚合与 EMA；MP 先跨 rank 合并同一 `(PID,modality)` 中心，再每行做一次 EMA。这些原型更新不进入本步或下一步的计算图，不是反向梯度路径。

## 4. 采样、训练状态和复现

原 PKM sampler 仍决定每 epoch 的 PID 顺序、样本 occurrence、group 耗尽和尾部丢弃。A1 只在其实际消耗的 group 中重排各 PID/模态样本，以优先让一个 K-group 覆盖更多相机；Epoch 1–10 不改变 PID 的 batch 安排。Epoch 10 后从完整训练划分提取固定身份中心，使用全局中心余弦与 TIR 涉及的模态中心余弦各占 0.5 建立对称困难矩阵。Epoch 11 起在同一 rank 内交换已有 PID-group：只接受两个 batch 仍各 PID 唯一、合计平均困难度正增益的交换，按 `softmax(Δ/0.2)` 抽取。交换和相机重组各有独立随机流；每 epoch 用账本断言 PID-group 次数及每模态样本 occurrence 次数不变。

MP bootstrap 和困难参考由 rank 0 使用未包装的模型做完整训练集确定性 forward，其他 rank 等待同步；不读取 Query/Gallery、不执行 backward 或 optimizer step。困难参考写入 `hard_reference_epoch010.pt` 并记录 SHA-256。训练状态 checkpoint 含模型、optimizer、scheduler、GradScaler、GPD/MP Bank、epoch、各 rank RNG、manifest 哈希和困难参考路径/哈希。恢复 Epoch 11 以后必须核验原参考，不重新提取。原 `transformer_*.pth` 推理权重格式保持不变。

AMP 发生跳步时，B0/A1 都不更新 GPD Bank，A1 也不更新 MP Bank。DDP 对成功步标志做 MIN/MAX 一致性检查，不一致则报错。B0 的其余 CE/WRT/ProCA/GPD 计算不变。

## 5. 启动示例

在 `WHU/` 目录下，单卡 A1：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/A1.yml MODEL.DIST_TRAIN False DATASETS.DATASET_DIR WHU-MARS SOLVER.IMS_PER_BATCH 64 SOLVER.BASE_LR 0.008 OUTPUT_DIR logs/A1_1000_single
```

四卡 A1（与原 README 的四卡 UAD 配对）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config_file configs/A1.yml MODEL.DIST_TRAIN True SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.032 OUTPUT_DIR logs/A1_1000_DDP
```

B0 使用同一组 GPU、batch、LR 和 seed，但配置文件改回 `configs/UAD.yml`，输出目录另设。恢复训练时给相应配置追加 `AV1.RESUME logs/.../training_state_epoch010.pt`；B0 也可从保存的训练状态恢复。测试仍按原 README 的 `test.py` 命令，权重指向 A1 的 `transformer_120.pth`，保持 `NECK_FEAT=before` 和 `FEAT_NORM=yes`。

## 6. 自检与尚需实机验证

- `python -m compileall -q WHU` 通过。
- 在 `WHU/` 运行 `python -m unittest discover -s tests -p test_av1.py -v`，5 项通过；新增测试明确确认同组不同模态图像 ID 不同、重复抽样同一原图 ID 相同、collate 顺序正确，以及 Mixed 掩码仅按相机规则与确切图像 ID 过滤。
- 解析两份 YAML 并去掉 A1 开关与输出目录后，确认 A1 的原 UAD 基础配置与 `UAD.yml` 完全相同。
- `processor` 导入和训练 collate 接口在本机通过。
- 本机 PyTorch 为 CPU 版，且没有 WHU-MARS 数据与 ImageNet 权重；因此尚未运行真实 bootstrap、CUDA AMP、NCCL DDP、120 epoch 训练和正式检索评测。增益与完整训练时间均未验证。
- 上机前建议先以真实数据跑 1 个 batch，查看各方向有效 Query、梯度有限性、Bank 覆盖与单卡/DDP 步数；随后跑完整 B0/A1 并比较总体和九种有向模态指标。Epoch 1、5 的加权梯度快照诊断需在真实有效 batch 上实施，当前只有合成梯度测试。

## 7. Epoch 10 服务器兼容性修正（2026-10-26）

服务器日志显示 Epoch 10 已写出 `hard_reference_epoch010.pt`，但随后各 rank 在 `torch.load(..., weights_only=False)` 处报 `TypeError: 'weights_only' is an invalid keyword argument for Unpickler()`。原因是服务器 PyTorch 版本较旧，不支持这个 `torch.load` 参数；该报错不说明困难矩阵生成失败。`WHU/processor/av1_state.py` 现提供 `load_trusted_artifact()`：仅对本训练代码自己生成的可信文件，根据当前 `torch.load` 签名决定是否传入 `weights_only=False`。训练恢复、Epoch 11 前参考加载和 Epoch 10 刚生成参考后的加载均改用它。新增旧版签名模拟测试；目前 6 项单元测试、静态编译通过。尚需在服务器真实版本上复跑验证。

`OUTPUT_DIR` 是日志和训练产物共用的目录：`train_log_*.txt` 是文本日志，`mp_bootstrap.pt` 是训练前 MP 原型，`hard_reference_epoch010.pt` 是 Epoch 10 固定困难参考；成功通过 Epoch 10 参考加载后才保存 `transformer_10.pth` 与 `training_state_epoch010.pt`。配置的常规 checkpoint 周期为 120，最终另保存 `transformer_120.pth` 与 `training_state_epoch120.pt`。当前所贴异常发生在 Epoch 10 checkpoint 保存之前，不能仅凭留下的 `hard_reference_epoch010.pt` 恢复模型、优化器与 Bank 状态。
