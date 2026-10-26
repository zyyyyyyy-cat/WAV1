# A-V1 vs UAD read-only retrieval diagnosis

All figures below come from the saved evaluation features. They describe associations, not causal effects of an individual training module.

## Official-protocol reproduction

| Model | Valid Q | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UAD | 6405 | 0.2971 | 0.4632 | 0.5455 | 0.1102 | 0.0066 |
| A1 | 6405 | 0.2765 | 0.4426 | 0.5252 | 0.1060 | 0.0064 |

## Rank transitions

- Rank-1: 472 regressed, 340 improved, net -132/6405 = -0.0206.
- Rank-5: 479 regressed, 347 improved, net -132/6405 = -0.0206.
- Rank-10: 470 regressed, 340 improved, net -130/6405 = -0.0203.
- 52 Queries regressed simultaneously at Rank-1/5/10; 470 crossed outward over Rank-10 and 340 crossed inward.

## AP and INP accounting by Rank-1 transition

- both_correct: n=1431, mean ΔAP=-0.008916, contribution to total ΔmAP=-0.001992, mean ΔINP=-0.000096.
- regressed: n=472, mean ΔAP=-0.047120, contribution to total ΔmAP=-0.003472, mean ΔINP=-0.000918.
- improved: n=340, mean ΔAP=+0.035838, contribution to total ΔmAP=+0.001902, mean ΔINP=+0.000626.
- both_wrong: n=4162, mean ΔAP=-0.000985, contribution to total ΔmAP=-0.000640, mean ΔINP=-0.000160.

## Closest positive and negative for Rank-1 regressions

- positive_decreased / negative_increased: 196 Queries.
- positive_decreased / negative_not_increased: 9 Queries.
- positive_not_decreased / negative_increased: 267 Queries.

## Mixed-gallery wrong top-1 sources

- TIR Ground → TIR Ground camera 1: 28 Queries.
- RGB Ground → RGB Ground camera 0: 20 Queries.
- RGB Ground → RGB Ground camera 2: 18 Queries.
- TIR Ground → TIR Ground camera 2: 18 Queries.
- TIR Ground → TIR Ground camera 0: 17 Queries.
- RGB Ground → NIR Ground camera 1: 13 Queries.
- NIR Ground → NIR Ground camera 1: 12 Queries.
- RGB Ground → RGB Ground camera 1: 12 Queries.
- NIR Ground → NIR Ground camera 0: 11 Queries.
- NIR Ground → NIR Ground camera 2: 11 Queries.
- NIR Ground → RGB Ground camera 1: 11 Queries.
- NIR Ground → NIR Ground camera 3: 9 Queries.
- NIR Ground → RGB Ground camera 0: 9 Queries.
- NIR Ground → RGB Ground camera 2: 9 Queries.
- RGB Ground → NIR Ground camera 2: 9 Queries.
- NIR Aerial → NIR Ground camera 3: 8 Queries.
- NIR Aerial → NIR Ground camera 4: 8 Queries.
- NIR Aerial → RGB Ground camera 0: 8 Queries.
- RGB Ground → NIR Aerial camera 5: 8 Queries.
- RGB Aerial → RGB Ground camera 3: 8 Queries.
- 189/472 Rank-1 regressions were correct in at least one single-modality Gallery but wrong in Mixed Gallery.

## Directed modality-pair retrieval

| Query → Gallery | Valid Q | ΔRank-1 | ΔmAP |
| --- | ---: | ---: | ---: |
| RGB → RGB | 2135 | -0.0300 | -0.0123 |
| RGB → NIR | 2135 | -0.0192 | -0.0036 |
| RGB → TIR | 2135 | -0.0033 | +0.0005 |
| RGB → Mixed | 2135 | -0.0197 | -0.0063 |
| NIR → RGB | 2135 | -0.0239 | -0.0024 |
| NIR → NIR | 2135 | -0.0215 | -0.0076 |
| NIR → TIR | 2135 | +0.0037 | +0.0008 |
| NIR → Mixed | 2135 | -0.0328 | -0.0036 |
| TIR → RGB | 2135 | +0.0098 | +0.0002 |
| TIR → NIR | 2135 | -0.0075 | -0.0012 |
| TIR → TIR | 2135 | -0.0117 | -0.0072 |
| TIR → Mixed | 2135 | -0.0094 | -0.0027 |

## Group counts and deltas

- query_modality=NIR: n=2135, ΔRank-1=-0.0328, ΔmAP=-0.0036, Δmargin=-0.01121450888747633.
- query_modality=RGB: n=2135, ΔRank-1=-0.0197, ΔmAP=-0.0063, Δmargin=-0.009555680857888429.
- query_modality=TIR: n=2135, ΔRank-1=-0.0094, ΔmAP=-0.0027, Δmargin=-0.009278334928480188.
- query_platform=Aerial: n=1350, ΔRank-1=-0.0193, ΔmAP=-0.0050, Δmargin=-0.009950037024639272.
- query_platform=Ground: n=5055, ΔRank-1=-0.0210, ΔmAP=-0.0040, Δmargin=-0.010033837823025905.
- modality_platform=NIR_Aerial: n=450, ΔRank-1=-0.0578, ΔmAP=-0.0087, Δmargin=-0.011225730412536198.
- modality_platform=NIR_Ground: n=1685, ΔRank-1=-0.0261, ΔmAP=-0.0023, Δmargin=-0.011211512041021172.
- modality_platform=RGB_Aerial: n=450, ΔRank-1=-0.0133, ΔmAP=-0.0070, Δmargin=-0.009852850569619072.
- modality_platform=RGB_Ground: n=1685, ΔRank-1=-0.0214, ΔmAP=-0.0061, Δmargin=-0.009476318026862442.
- modality_platform=TIR_Aerial: n=450, ΔRank-1=+0.0133, ΔmAP=+0.0007, Δmargin=-0.008771530091762543.
- modality_platform=TIR_Ground: n=1685, ΔRank-1=-0.0154, ΔmAP=-0.0036, Δmargin=-0.0094136834011941.
- uad_first_rank=1: n=1903, ΔRank-1=-0.2480, ΔmAP=-0.0184, Δmargin=-0.015618410538950533.
- uad_first_rank=11-20: n=535, ΔRank-1=+0.0411, ΔmAP=+0.0043, Δmargin=-0.00303055265239466.
- uad_first_rank=2-5: n=1064, ΔRank-1=+0.2453, ΔmAP=+0.0004, Δmargin=-0.004155447356458893.
- uad_first_rank=21-50: n=717, ΔRank-1=+0.0084, ΔmAP=+0.0023, Δmargin=-0.005669306698703367.
- uad_first_rank=51-100: n=500, ΔRank-1=+0.0020, ΔmAP=+0.0025, Δmargin=-0.005299160957336426.
- uad_first_rank=6-10: n=527, ΔRank-1=+0.0930, ΔmAP=+0.0029, Δmargin=-0.0056011615713362.
- uad_first_rank=>100: n=1159, ΔRank-1=+0.0009, ΔmAP=+0.0008, Δmargin=-0.016154217201898177.

Interpretation should use the per-Query CSVs and subgroup counts. A sign of similarity change alone does not establish why a module caused it.
