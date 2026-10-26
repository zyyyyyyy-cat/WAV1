# MINER: Modeling Inter- and Intra-Person Variance for Re-Identification
Oron Nir, Eliyahu Strugo, Ariel Shamir

## Abstract
Clothes-Changing Person Re-Identification (CC-ReID) is a long-standing and challenging task due to drastic appearance shifts across outfits. 
Most prior works devise biometric signal extraction heuristics to enhance inter-person variance, e.g., using segmentation masks, while the intra-person variance is usually left uncontrolled.
We present MINER, a model-agnostic framework to directly contrast hard examples and suppress garment biases.
Extensive experiments show that controlled simultaneous intra- and inter-person variance outperforms the state of the art on seven CC-ReID and Visible-Infrared ReID datasets.
We further demonstrate the importance of balancing these factors, which vary with the dataset priors.


# MINER Pipeline for ICPR 2026 TVRID RGB-RGB Track

## Step 0: Pre-compute Embeddings (one-time)

```
PRCC Checkpoint ──→ EVA02-L ──→ all 20,432 training frames
(eva02_l_bio_best.pth)         ──→ tvrid_embeddings.json
                                    {path: [1024-dim], ...}
```

Script: `python generate_embeddings.py`

---

## Step 1: Build Centroids (at sampler init)

From the pre-computed embeddings, build two index structures:

```
tvrid_embeddings.json
     │
     ├──→ Per-PID centroids:        mean(embeddings) per PID
     │    [62 PIDs × 1024-dim]      → centroid matrix
     │
     └──→ Per-PID-per-camera:       mean(embeddings) per (PID, camera) pair
          e.g. PID 5: {flat: c₀, upward: c₁, downward: c₂}
          → cam_centroids dict
```

---

## Step 2: MINER Batch Construction (Algorithm 1, every iteration)

Hyperparameters: **m** = IDS_PER_BATCH, **nq** = **ng** = REPS_PER_ID.

Note: In the paper, Algorithm 1 first samples a camera pair (cq, cg),
then builds the candidate PID set from identities present in both cameras.
In TVRID all 62 PIDs appear in all 4 cameras, so we skip the camera-pair
filtering and mine PIDs from the full set, then select the hardest camera
pair per-PID.

### 2a. Hard PID Selection (inter-person mining)

```
Pick seed PID uniformly at random
    │
    ▼
Cosine similarity: seed centroid vs all 62 PID centroids
    │
    ▼
Softmax(similarities / τ_mining) → sampling probabilities
    │
    ▼
Multinomial sample (m-1) more PIDs (without replacement)
    │
    ▼
m visually-similar PIDs per batch (hard negatives)
```

### 2b. Hard Camera Pair (intra-person mining, per PID)

In CC-ReID, MINER mines over *clothing pairs* (Eq. 4-7 in the paper).
TVRID has no clothing changes, so we use **camera angle** as the
appearance factor instead (analogous to modality in VI-ReID, Section 5.4).

```
For each of m PIDs:
    │
    ▼
Compare all camera-pair centroids:
    cos(flat, upward), cos(flat, downward), cos(upward, downward), ...
    │
    ▼
Pick pair with LOWEST cosine similarity (most dissimilar views)
    e.g. flat ↔ downward
    │
    ▼
Randomly assign one as cam_q, other as cam_g
```

Note: The paper samples clothing pairs stochastically (Eq. 6-7).
For TVRID we use deterministic argmin since each PID has only
4 camera angles (6 possible pairs), making stochastic sampling
unnecessary.

### 2c. Frame Sampling

```
For each PID:
    ├──→ Sample nq frames from cam_q
    └──→ Sample ng frames from cam_g

Batch: [q₁ ... q_{m·nq} | g₁ ... g_{m·ng}]    (2·m·nq frames total)
        ←── m × nq ───→   ←── m × ng ───→
```

---

## Step 3: Forward Pass + MINER Loss

Let N = m·nq (= m·ng).

```
2N images ──→ EVA02-L backbone ──→ 2N × 1024-dim features
                                         │
                       ┌─────────────────┤
                       ▼                 ▼
                 feat_q [N×1024]   feat_g [N×1024]
                       │                 │
                       ▼                 ▼
                 L2-normalize      L2-normalize
                       │                 │
                       └────────┬────────┘
                                ▼
                 S = feat_q @ feat_gᵀ / τ     [N × N]
```

### Block-Diagonal Target Matrix Y

The target matrix Y has a **block-diagonal** structure — each block
corresponds to one PID's query-gallery positive pairs:

```
Example with m=4 PIDs, nq=ng=3 (N=12):

           g₁  g₂  g₃  g₄  g₅  g₆  g₇  g₈  g₉  g₁₀ g₁₁ g₁₂
         ┌───────────┬───────────┬───────────┬───────────┐
   q₁    │  1  1  1  │  0  0  0  │  0  0  0  │  0  0  0  │
   q₂    │  1  1  1  │  0  0  0  │  0  0  0  │  0  0  0  │
   q₃    │  1  1  1  │  0  0  0  │  0  0  0  │  0  0  0  │
         ├───────────┼───────────┼───────────┼───────────┤
   q₄    │  0  0  0  │  1  1  1  │  0  0  0  │  0  0  0  │
   q₅    │  0  0  0  │  1  1  1  │  0  0  0  │  0  0  0  │
   q₆    │  0  0  0  │  1  1  1  │  0  0  0  │  0  0  0  │
         ├───────────┼───────────┼───────────┼───────────┤
   q₇    │  0  0  0  │  0  0  0  │  1  1  1  │  0  0  0  │
   q₈    │  0  0  0  │  0  0  0  │  1  1  1  │  0  0  0  │
   q₉    │  0  0  0  │  0  0  0  │  1  1  1  │  0  0  0  │
         ├───────────┼───────────┼───────────┼───────────┤
   q₁₀   │  0  0  0  │  0  0  0  │  0  0  0  │  1  1  1  │
   q₁₁   │  0  0  0  │  0  0  0  │  0  0  0  │  1  1  1  │
   q₁₂   │  0  0  0  │  0  0  0  │  0  0  0  │  1  1  1  │
         └───────────┴───────────┴───────────┴───────────┘

   1 = positive (same PID, cross-camera)
   0 = negative (different PID, hard-mined)

   m blocks of nq × ng along the diagonal
```

### Loss Computation (Equations 9-11)

```
L_q→g:  for each row of S, softmax over columns,
         target = row of Y normalized to sum to 1        (Eq. 9)

L_g→q:  for each column of S, softmax over rows,
         target = column of Y normalized to sum to 1     (Eq. 10)

L_total = 0.5 × (L_q→g + L_g→q)                         (Eq. 11)
```

Training: 2 epochs, SGD, cosine LR with linear warmup.

---

## Step 4: Submission

Let k = EVAL_FRAMES (default 5).

```
For each of 204 test passages:
    k evenly-spaced frames ──→ EVA02-L ──→ mean-pool ──→ 1024-dim

204 passage embeddings ──→ pairwise cosine distance [204 × 204]
                       ──→ rank by distance per query
                       ──→ rankings_rgb.csv
```

Optional flags in `submit.py`:
- `--trim-pct P` — skip the first/last P% of frames per passage
  (removes background-only frames at passage boundaries)
- `--rerank` — apply k-reciprocal re-ranking (Zhong et al. 2017)
  with `--k1`, `--k2`, `--lambda-value` parameters
