# A-V1 免训练检索诊断

本工具只读取已经训练完成的 UAD 与 A-V1 权重，以及同一份 WHU-MARS-1000 Query/Gallery。它调用原模型的测试前向获取最终 LayerNorm 后、BNNeck 前的 768 维 CLS，进行 L2 归一化；采用与 `utils/metrics.py` 相同的平方欧氏距离、完整排序、排除全部同相机 Gallery 的协议。模型始终处于 `eval()` 与 `torch.inference_mode()`；不加载训练 Bank、不做 backward、不更新模型或 BN。

## 运行

在 `WHU/` 目录执行，路径按服务器实际文件调整：

```bash
python tools/diagnose_av1_retrieval.py \
  --config-file configs/UAD.yml \
  --uad logs/UAD_1000_DDP/transformer_120.pth \
  --a1 logs/A1_1000_DDP/transformer_120.pth \
  --output logs/diagnosis_1000 \
  --data-root /project/code/data \
  --dataset-dir WHU-MARS \
  --batch-size 128 --workers 4 --query-chunk 16
```

默认用指导文档中的原 UAD（mAP 11.02%、R1 29.71%、R5 46.32%、R10 54.55%）与 A1（10.60%、27.65%、44.26%、52.52%）检查官方总体指标，允许绝对误差 `0.0002`。如果相应日志的精确结果不同，可分别覆盖：

```bash
--expected-uad 'mAP=0.1102,rank1=0.2971,rank5=0.4632,rank10=0.5455' \
--expected-a1 'mAP=0.1060,rank1=0.2765,rank5=0.4426,rank10=0.5252'
```

指标不匹配时程序报错并停止归因分析，应先检查权重、数据划分、相机 ID、归一化与评测配置。工具要求 `FILTER_GD=False`、`TOP_K_EVAL=0`、关闭 rerank，且按 RGB/IR/Thermal 顺序输入。首次运行抽取特征；后续可加 `--reuse-cache`，但只在 checkpoint SHA-256 和完整图像 manifest SHA-256 均匹配时复用。

若两种方法在 Epoch 50、60 等均已有 checkpoint，可重复添加：

```bash
--checkpoint-pair 50 logs/UAD_1000_DDP/transformer_50.pth logs/A1_1000_DDP/transformer_50.pth \
--checkpoint-pair 60 logs/UAD_1000_DDP/transformer_60.pth logs/A1_1000_DDP/transformer_60.pth
```

无现成中间权重时不要添加这部分。脚本不从训练日志推造逐 Query 的中间结果。

## 输出

| 文件 | 内容 |
| --- | --- |
| `UAD/`、`A1/` 下的 `query_features.npy`、`gallery_features.npy` | 已按官方顺序保存的 L2 pre-BN CLS 特征。 |
| `query_metadata.csv`、`gallery_metadata.csv` | 稳定相对图像路径、PID、相机、模态、平台和官方集合索引。 |
| `diagnostic_manifest.json` | checkpoint/manifest 哈希、输入、预处理、数量和协议。 |
| `retrieval_queries.csv` | 每个有效 Query 的两模型排名、AP、INP、正负最近邻相似度、margin 与差值。 |
| `excluded_queries.csv` | 因无跨相机正确 Gallery 被官方规则排除的 Query。 |
| `query_rank_transitions.csv` | Rank-1/5/10 四类转变和首个正确排名分桶转移。 |
| `topk_gallery.csv` | 两模型各 Query 的完整有效候选排序前 20 名；可用 `--top-k` 修改，至少为 10。 |
| `regressed_queries.csv` | 原 UAD Rank-1 正确、A1 错误的 Query，包含最危险错误 Gallery 的图像 ID。 |
| `modality_competition.csv`、`modality_pair_summary.csv` | 每个 Query 在 RGB、NIR、TIR 与 Mixed Gallery 的竞争及九个有向模态组合汇总。 |
| `group_comparison.csv` | 按 Query 模态/平台/相机、正样本数量、UAD 原始难度和 A1 错误来源分组。 |
| `feature_geometry_summary.csv` | 正负、同/跨模态、同/跨平台相似度的分位数近似；直方图分辨率 0.005。 |
| `ranking_summary.json`、`diagnostic_report.md` | 指标、转变、贡献分解、错误来源和描述性结论。 |
| `checkpoint_dynamics.csv` | 仅在提供中间 checkpoint 对时生成。 |

逐 Query 排名、AP、INP 均使用全量有效 Gallery；分布分位数通过有限宽直方图近似。总体 Mixed 指标不由九种模态任务平均得到。所有单模态任务独立筛选 Gallery，仅对该任务中有有效正确 Gallery 的 Query 计算。

## 正确性边界

该工具不会自动判断某个损失或采样器是退化原因；报告只描述观测到的 Query 转变、相似度变化与错误来源。若权重或数据未提供，脚本无法生成真实统计。当前本地没有这两套 checkpoint 或 WHU-MARS-1000 数据，因此已验证的是静态编译、合成排序与正式 `eval_func` 一致性、特征与图像清单对齐检查，以及统计聚合恒等式；服务器完整运行仍需实际执行。
