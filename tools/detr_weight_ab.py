"""A/B: does applying DETR's `weight_dict` stop the query collapse?

Arm A sets every weight to 1.0, reproducing the pre-fix objective exactly
(`engine.train_one_epoch` summed the raw components the criterion returned).
Arm B uses the real DETR weighting (ce 1 / bbox 5 / giou 2). Same seed, same
data, same schedule — only the weighting differs.

This is a *mechanism check* for the fix in `presets.SetCriterion.forward`, not
the real retrain (that is `configs/models/torchvision_detr.yaml`): it runs over a
small subset at a reduced resolution for a handful of epochs. It picks CUDA when
available and falls back to CPU.

    python tools/detr_weight_ab.py                      # ~15 epochs per arm
    python tools/detr_weight_ab.py --epochs 1 --train 32 --val 32   # wiring check
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fabric_defect_hub.datasets.zju_leaper import ZJULeaperDataset
from fabric_defect_hub.models.torchvision.dataset import SampleDetectionDataset, detection_collate_fn
from fabric_defect_hub.models.torchvision.engine import evaluate, run_training
from fabric_defect_hub.models.torchvision.presets import build_model

PATTERNS = ["pattern1", "pattern2", "pattern3", "pattern4"]
N_TRAIN, N_VAL, EPOCHS, BATCH = 240, 96, 15, 8
MIN_SIZE, MAX_SIZE = 256, 320
ARMS = {
    "A_all_ones (pre-fix)": {"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
    "B_detr_weights (fixed)": {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
}
OUT_PATH = Path("runs/detr_weight_ab.json")


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_samples(seed: int):
    train = ZJULeaperDataset(root="datasets/textile/ZJU-Leaper", split="train", pattern=PATTERNS,
                             num_samples=N_TRAIN, use_defect=True, defect_ratio=1.0,
                             task="detection", seed=seed).load_samples()
    val = ZJULeaperDataset(root="datasets/textile/ZJU-Leaper", split="test", pattern=PATTERNS,
                           num_samples=N_VAL, use_defect=True, defect_ratio=1.0,
                           task="detection", seed=seed).load_samples()
    return [s for s in train if s.annotations.boxes], [s for s in val if s.annotations.boxes]


def probe(model, samples, device) -> dict[str, float]:
    """Collapse diagnostics: do predictions vary across the 100 queries, and
    across different images, or is the model emitting one constant box?"""

    loader = DataLoader(SampleDetectionDataset(samples, class_map={"defect": 1}),
                        batch_size=8, collate_fn=detection_collate_fn)
    boxes, scores = [], []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            for out in model([image.to(device) for image in images]):
                boxes.append(out["boxes"].detach().cpu())
                scores.append(out["scores"].detach().cpu())
    b, s = torch.stack(boxes), torch.stack(scores)
    return {
        "query_spread": float(b.std(dim=1).mean()),               # across the 100 queries
        "image_spread": float(b.mean(dim=1).std(dim=0).mean()),   # across images
        "top_score": float(s.max(dim=1).values.mean()),
        "n_pred_over_0.5": float((s > 0.5).sum(dim=1).float().mean()),
    }


def run_arm(name: str, weight_dict: dict, train_samples, val_samples, device):
    print(f"\n{'=' * 70}\nARM {name}: weight_dict={weight_dict}\n{'=' * 70}", flush=True)
    seed_all(0)
    model = build_model("detr_resnet50", num_classes=2, pretrained=True,
                        trainable_backbone_layers=3, min_size=MIN_SIZE, max_size=MAX_SIZE)
    model.criterion.weight_dict = dict(weight_dict)

    train_loader = DataLoader(SampleDetectionDataset(train_samples, class_map={"defect": 1}),
                              batch_size=BATCH, shuffle=True, num_workers=2,
                              generator=torch.Generator().manual_seed(0),
                              collate_fn=detection_collate_fn)
    val_loader = DataLoader(SampleDetectionDataset(val_samples, class_map={"defect": 1}),
                            batch_size=BATCH, shuffle=False, num_workers=2,
                            collate_fn=detection_collate_fn)

    history: list[dict] = []

    def on_epoch_end(log, *_args):
        history.append({"epoch": log.epoch, "train_loss": log.train_loss,
                        "map": log.val_metrics.get("map"), "map_50": log.val_metrics.get("map_50")})
        print(f"  epoch {log.epoch:>2}: loss {log.train_loss:.4f} | "
              f"map {log.val_metrics.get('map', float('nan')):.5f} | "
              f"map_50 {log.val_metrics.get('map_50', float('nan')):.5f}", flush=True)

    started = time.time()
    _, best_map = run_training(
        model, train_loader, val_loader, device, epochs=EPOCHS,
        optimizer_name="adamw", lr=1e-4, momentum=0.9, weight_decay=5e-4,
        lr_scheduler_name="cosine", step_size=10, gamma=0.1, warmup_epochs=1,
        grad_clip_norm=5.0, patience=0, with_masks=False, amp=False,
        backbone_lr=1e-5, resume_state=None, on_epoch_end=on_epoch_end, task="detect",
    )
    final = evaluate(model, val_loader, device)
    diagnostics = probe(model, val_samples[:32], device)
    result = {"arm": name, "weight_dict": weight_dict, "best_map": best_map,
              "final_map": final.get("map"), "final_map_50": final.get("map_50"),
              "diagnostics": diagnostics, "minutes": round((time.time() - started) / 60, 1),
              "history": history}
    print(f"  -> best_map {best_map:.5f} | final map {final.get('map'):.5f} | "
          f"{diagnostics} | {result['minutes']} min", flush=True)
    return result


def main() -> None:
    global N_TRAIN, N_VAL, EPOCHS, BATCH, MIN_SIZE, MAX_SIZE, OUT_PATH

    import argparse

    parser = argparse.ArgumentParser(description="DETR loss-weighting A/B.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train", type=int, default=N_TRAIN)
    parser.add_argument("--val", type=int, default=N_VAL)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--size", type=int, default=MIN_SIZE)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    N_TRAIN, N_VAL, EPOCHS, BATCH = args.train, args.val, args.epochs, args.batch
    MIN_SIZE, MAX_SIZE = args.size, args.size + 64
    OUT_PATH = Path(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} threads={torch.get_num_threads()} "
          f"train={N_TRAIN} val={N_VAL} epochs={EPOCHS} size={MIN_SIZE}", flush=True)
    train_samples, val_samples = load_samples(0)
    results = [run_arm(name, weights, train_samples, val_samples, device)
               for name, weights in ARMS.items()]

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for result in results:
        diag = result["diagnostics"]
        print(f"{result['arm']:>24}: best_map={result['best_map']:.5f} "
              f"final_map={result['final_map']:.5f} "
              f"query_spread={diag['query_spread']:.5f} "
              f"image_spread={diag['image_spread']:.5f} "
              f"top_score={diag['top_score']:.3f}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
