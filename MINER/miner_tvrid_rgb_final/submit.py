"""Generate TVRID RGB track submission rankings.

Produces ranking CSVs for the TVRID challenge (RGB track). For each test
passage (query), all other passages are ranked by cosine distance of
passage-level embeddings (mean of N frame features).

Usage:
    python submit.py --config configs/rgb.yml \
        --checkpoint path/to/model.pth \
        --output submission/
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from config import TVRID_NUM_CAMERAS, TVRID_DEFAULT_NUM_CLASSES, EVA02_EMBED_DIM
from model import build_model
from transforms import build_transforms
from dataset import _list_frames, _select_frames_evenly
from utils import load_checkpoint
from evaluation import re_ranking


def embed_passage(model, frame_paths, transform, device) -> torch.Tensor:
    """Embed a single passage by mean-pooling frame-level features.

    Returns L2-normalized 1D embedding tensor on CPU.
    """
    imgs = [transform(Image.open(p).convert("RGB")) for p in frame_paths]
    batch = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = model(batch)
    avg_feat = F.normalize(feats.cpu().mean(dim=0, keepdim=True), p=2, dim=1)
    return avg_feat.squeeze(0)


def generate_rankings(gallery_ids, paths, dist_matrix):
    """Build ranked lists from a distance matrix."""
    results = []
    for i in tqdm(range(dist_matrix.shape[0]), desc="Ranking"):
        qid = gallery_ids[i]
        row = dist_matrix[i]
        rank = 1
        for g_idx in torch.argsort(row).tolist():
            gid = gallery_ids[g_idx]
            if gid == qid:
                continue
            results.append({
                "query_gallery_id": qid,
                "query_path": paths[i],
                "gallery_id": gid,
                "gallery_path": paths[g_idx],
                "rank": rank,
                "distance": float(row[g_idx].item()),
            })
            rank += 1
    return results


def get_args():
    parser = argparse.ArgumentParser(description="Generate TVRID RGB submission rankings")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth model checkpoint")
    parser.add_argument("--output", default="submission", help="Output directory for CSV files")
    parser.add_argument("--data-root", default=None, help="Override DATA.ROOT from config")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Frames per passage (overrides config EVAL_FRAMES)")
    parser.add_argument("--trim-pct", type=float, default=0,
                        help="Skip first/last N%% of frames per passage")
    parser.add_argument("--rerank", action="store_true",
                        help="Apply k-reciprocal re-ranking (Zhong et al. 2017)")
    parser.add_argument("--k1", type=int, default=20, help="Re-ranking k1")
    parser.add_argument("--k2", type=int, default=6, help="Re-ranking k2")
    parser.add_argument("--lambda-value", type=float, default=0.3, help="Re-ranking lambda")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.data_root is not None:
        cfg.setdefault("DATA", {})["ROOT"] = args.data_root

    data_root = cfg.get("DATA", {}).get("ROOT", "./data")
    db_dir = os.path.join(data_root, "DB_extracted")
    test_csv = os.path.join(db_dir, "public_test_labels.csv")
    test_base = os.path.join(db_dir, "test_public")

    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    num_frames = args.num_frames or cfg.get("DATA", {}).get("EVAL_FRAMES", 5)

    # Build model — infer num_classes from checkpoint
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    clean = {k.replace("module.", ""): v for k, v in state_dict.items()}
    num_classes = clean["head.weight"].shape[0] if "head.weight" in clean else TVRID_DEFAULT_NUM_CLASSES

    model = build_model(num_classes=num_classes, num_cameras=TVRID_NUM_CAMERAS)
    load_checkpoint(model, args.checkpoint)
    model.to(args.device)
    model.eval()

    # Read test passages
    print(f"Reading test passages from: {test_csv}")
    passages = []
    with open(test_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            passages.append({
                "gallery_id": str(row["gallery_id"]),
                "path": row["path"],
                "passage_dir": os.path.join(test_base, row["path"].replace("\\", "/")),
            })
    print(f"Found {len(passages)} test passages")

    _, transform_test = build_transforms(cfg)

    _MIN_FRAMES_FOR_TRIM = 5  # don't trim passages shorter than this

    def _trim_frames(frames, pct):
        if pct <= 0 or len(frames) < _MIN_FRAMES_FOR_TRIM:
            return frames
        skip = max(1, int(len(frames) * pct / 100))
        return frames[skip : len(frames) - skip]

    # Embed all passages
    print(f"\nEmbedding passages ({num_frames} frames each)...")
    embeddings = []
    for passage in tqdm(passages, desc="Embedding"):
        frames = _list_frames(passage["passage_dir"])
        frames = _trim_frames(frames, args.trim_pct)
        if num_frames > 0:
            frames = _select_frames_evenly(frames, num_frames)
        if not frames:
            print(f"WARNING: No RGB frames in {passage['passage_dir']}")
            embeddings.append(torch.zeros(EVA02_EMBED_DIM))
        else:
            embeddings.append(embed_passage(model, frames, transform_test, args.device))

    emb_mat = torch.stack(embeddings)

    # Compute distances
    print("Computing pairwise distances...")
    if args.rerank:
        print(f"  Re-ranking (k1={args.k1}, k2={args.k2}, lambda={args.lambda_value})...")
        dist_mat = torch.tensor(
            re_ranking(emb_mat, emb_mat, k1=args.k1, k2=args.k2, lambda_value=args.lambda_value),
            dtype=torch.float32,
        )
    else:
        sim = F.cosine_similarity(emb_mat.unsqueeze(1), emb_mat.unsqueeze(0), dim=2)
        dist_mat = 1.0 - sim

    gallery_ids = [p["gallery_id"] for p in passages]
    paths = [p["path"] for p in passages]
    results = generate_rankings(gallery_ids, paths, dist_mat)

    # Write output
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, "rankings_rgb.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query_gallery_id", "query_path", "gallery_id", "gallery_path", "rank", "distance",
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows to {output_file}")
    print(f"  {len(passages)} queries x {len(passages) - 1} gallery entries each")


if __name__ == "__main__":
    main()
