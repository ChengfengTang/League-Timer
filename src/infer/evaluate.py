"""Score predicted ability events against hand labels.

Matches predictions to ground-truth events of the same ability within a time
tolerance (greedy, highest-confidence first) and reports event-level
precision / recall / F1 per ability plus a micro-average.

Run::

    python -m src.infer.evaluate --config configs/{ChampionName}.yaml \
        --pred outputs/test.events.json \
        --truth data/{ChampionName}/annotations/test.json \
        --tolerance 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.common.annotations import Annotation
from src.common.config import Config


def _load_pred(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("events", data if isinstance(data, list) else [])


def match_ability(preds: List[Dict], truth_times: List[float], tolerance: float):
    """Greedy match for a single ability. Returns (tp, fp, fn)."""
    preds = sorted(preds, key=lambda e: e.get("score", 1.0), reverse=True)
    remaining = sorted(truth_times)
    matched = [False] * len(remaining)
    tp = 0
    for p in preds:
        best_j = -1
        best_d = tolerance + 1e-9
        for j, t in enumerate(remaining):
            if matched[j]:
                continue
            d = abs(t - p["time"])
            if d <= tolerance and d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            matched[best_j] = True
            tp += 1
    fp = len(preds) - tp
    fn = matched.count(False)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def run(config_path: str, pred_path: str, truth_path: str, tolerance: float) -> None:
    cfg = Config.load(config_path)
    preds = _load_pred(pred_path)
    truth = Annotation.load(truth_path)

    abilities = cfg.ability_classes
    print(f"Tolerance: +/-{tolerance:.2f}s\n")
    print(f"  {'ability':<10} {'TP':>4} {'FP':>4} {'FN':>4} {'prec':>6} {'rec':>6} {'f1':>6}")

    tot_tp = tot_fp = tot_fn = 0
    for ab in abilities:
        ab_preds = [e for e in preds if e["ability"] == ab]
        ab_truth = [e.time for e in truth.events if e.ability == ab]
        tp, fp, fn = match_ability(ab_preds, ab_truth, tolerance)
        p, r, f1 = prf(tp, fp, fn)
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        print(f"  {ab:<10} {tp:4d} {fp:4d} {fn:4d} {p:6.3f} {r:6.3f} {f1:6.3f}")

    p, r, f1 = prf(tot_tp, tot_fp, tot_fn)
    print(f"  {'-' * 44}")
    print(f"  {'micro':<10} {tot_tp:4d} {tot_fp:4d} {tot_fn:4d} {p:6.3f} {r:6.3f} {f1:6.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Event-level evaluation of predictions vs labels.")
    p.add_argument("--config", required=True)
    p.add_argument("--pred", required=True, help="events.json from the recognizer")
    p.add_argument("--truth", required=True, help="annotation JSON for the same video")
    p.add_argument("--tolerance", type=float, default=0.5, help="match tolerance in seconds")
    args = p.parse_args()
    run(args.config, args.pred, args.truth, args.tolerance)


if __name__ == "__main__":
    main()
