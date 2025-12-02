#!/usr/bin/env python3
"""
Evaluate performances of handwriting decoders (baseline and set-transformer) with/without augmentation.

For each provided checkpoint, runs test under:
  1) clean (no augmentation)
  2) rotation (random channel rotation; k in [-8, 8])
  3) permutation (random channel permutation; rotation disabled)

Each condition is repeated 10 times with distinct seeds, and results are aggregated.

Outputs:
  - eval_results.csv      (per-run metrics: model, condition, seed, test/CER, test_loss, etc.)
  - eval_results.json     (same, JSON)
  - eval_summary.csv      (aggregated stats per model x condition: mean/std/min/max for key metrics)

Usage example:
python eval_results.py \
  --data_dir ~/emg_data \
  --config_yaml ~/emg_models/handwriting/model_config.yaml \  # fallback default
  --repeats 10 \
  --baseline_noaug_ckpt /runs/baseline_noaug/best.ckpt \
  --baseline_rot_ckpt   /runs/baseline_trainrot/best.ckpt \
  --set_noaug_ckpt      /runs/set_noaug/best.ckpt \
  --set_rot_ckpt        /runs/set_trainrot/best.ckpt \
  --baseline_noaug_yaml /runs/baseline_noaug/.hydra/config.yaml \
  --baseline_rot_yaml   /runs/baseline_trainrot/.hydra/config.yaml \
  --set_noaug_yaml      /runs/set_noaug/.hydra/config.yaml \
  --set_rot_yaml        /runs/set_trainrot/.hydra/config.yaml
"""

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from hydra.utils import instantiate
from generic_neuromotor_interface.lightning import HandwritingModule
from generic_neuromotor_interface.utils import handwriting_collate
from torch.utils.data import DataLoader


def _make_aug_collate(rotation: Optional[Dict[str, Any]] = None,
                      permutation: Optional[Dict[str, Any]] = None,
                      seed: int = 0):
    """
    Wraps handwriting_collate so we can apply channel rotation/permutation to batch['emg'] (N,T,C).
    """
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    def _collate(samples):
        batch = handwriting_collate(samples)
        x = batch["emg"]  # (N, T, C)
        C = int(x.shape[-1])

        # Rotation
        if rotation and rotation.get("enable", False):
            if rotation.get("random_k", False):
                lo, hi = rotation.get("k_range", [-8, 8])
                k = int(torch.randint(low=int(lo), high=int(hi) + 1, size=(1,), generator=gen))
            else:
                k = int(rotation.get("k", 0))
            # normalize to shortest equivalent rotation
            k = ((k % C) + C) % C
            if k > C // 2:
                k -= C
            x = torch.roll(x, shifts=k, dims=-1)

        # Permutation
        if permutation and permutation.get("enable", False):
            if permutation.get("mode", "random") == "random":
                perm = torch.randperm(C, generator=gen, device=x.device)
            else:
                perm = torch.as_tensor(permutation["perm"], dtype=torch.long, device=x.device)
            x = x.index_select(-1, perm)

        batch["emg"] = x
        return batch

    return _collate


# compose cfg with data_location and corpus CSV
def _cfg_with_data(cfg: Dict, data_dir: str) -> Dict:
    c = copy.deepcopy(cfg)
    c["data_module"]["data_location"] = os.path.expanduser(data_dir)
    dm = c["data_module"]
    # If split is from_csv, set the csv path
    try:
        tgt = dm["data_split"]["_target_"]
        if isinstance(tgt, str) and "from_csv" in tgt:
            dm["data_split"]["csv_filename"] = os.path.join(
                c["data_module"]["data_location"], "handwriting_corpus.csv"
            )
    except Exception:
        pass
    return c


# eval one model under a given aug, 10 repeats
def evaluate_model(ckpt_path: str,
                   base_cfg: Dict,
                   test_aug: Optional[Dict[str, Any]],
                   repeats: int,
                   base_seed: int = 1000) -> List[Dict[str, Any]]:
    """
    Returns a list of per-run metrics dictionaries (including 'seed' and 'condition').
    """
    rows = []

    for r in range(repeats):
        seed = base_seed + r

        # try DataModule with test channel_aug (preferred)
        dm_cfg = copy.deepcopy(base_cfg["data_module"])
        use_fallback = False

        # set per-repeat seed if the DataModule supports it
        if "seed" in dm_cfg:
            dm_cfg["seed"] = seed

        if test_aug is not None:
            dm_cfg.setdefault("channel_aug", {})
            dm_cfg["channel_aug"]["test"] = test_aug

        try:
            datamodule = instantiate(dm_cfg)
        except TypeError:
            datamodule = instantiate(base_cfg["data_module"])
            use_fallback = True

        # load model
        model = HandwritingModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
        model.eval()

        # build trainer
        trainer = Trainer(accelerator="auto", devices="auto", logger=False, enable_progress_bar=True)

        if not use_fallback or test_aug is None:
            # clean path or channel_aug supported by DataModule
            out = trainer.test(model=model, datamodule=datamodule)
        else:
            # fallback path
            datamodule.setup("test")
            base_loader = datamodule.test_dataloader()

            rotation = test_aug.get("rotation") if test_aug else None
            permutation = test_aug.get("permutation") if test_aug else None

            aug_collate = _make_aug_collate(rotation=rotation, permutation=permutation, seed=seed)
            aug_loader = DataLoader(
                datamodule.test_dataset,
                batch_size=base_loader.batch_size,
                shuffle=False,
                num_workers=base_loader.num_workers,
                pin_memory=True,
                collate_fn=aug_collate,
            )
            out = trainer.test(model=model, dataloaders=aug_loader)

        res = out[0] if out else {}
        # normalize keys (some runs may use slightly different metric names)
        cer = None
        for k in ["test/CER", "test_character_error_rate", "character_error_rate", "cer"]:
            if k in res:
                cer = float(res[k]); break
        loss = float(res.get("test_loss", np.nan))

        res_row = dict(res)
        res_row.update({"seed": seed, "cer": cer, "loss": loss})
        rows.append(res_row)

    return rows


def summarize(rows: List[Dict[str, Any]], metrics=("cer", "loss")) -> Dict[str, Any]:
    summary = {}
    for m in metrics:
        vals = [r[m] for r in rows if r.get(m) is not None and not np.isnan(r[m])]
        if not vals:
            summary[m] = {"mean": None, "std": None, "min": None, "max": None}
        else:
            arr = np.array(vals, dtype=float)
            summary[m] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder containing handwriting_corpus.csv and HDF5 files")
    ap.add_argument("--config_yaml", required=True, help="Default merged config YAML (used if per-model YAML not given)")
    ap.add_argument("--repeats", type=int, default=10, help="Number of repeats per condition")
    ap.add_argument("--baseline_noaug_ckpt", default=None)
    ap.add_argument("--baseline_rot_ckpt", default=None)
    ap.add_argument("--set_noaug_ckpt", default=None)
    ap.add_argument("--set_rot_ckpt", default=None)
    ap.add_argument("--baseline_noaug_yaml", default=None, help="Merged YAML for baseline_noaug (overrides --config_yaml)")
    ap.add_argument("--baseline_rot_yaml",   default=None, help="Merged YAML for baseline_trainrot (overrides --config_yaml)")
    ap.add_argument("--set_noaug_yaml",      default=None, help="Merged YAML for set_noaug (overrides --config_yaml)")
    ap.add_argument("--set_rot_yaml",        default=None, help="Merged YAML for set_trainrot (overrides --config_yaml)")
    ap.add_argument("--out_prefix", default="eval_results", help="Prefix for output files")
    args = ap.parse_args()

    # # load config and set data location / corpus CSV
    # cfg = OmegaConf.load(os.path.expanduser(args.config_yaml))
    # cfg = _cfg_with_data(cfg, args.data_dir)
    def load_cfg(yaml_path: str):
        cfg_ = OmegaConf.load(os.path.expanduser(yaml_path))
        return _cfg_with_data(cfg_, args.data_dir)

    global_cfg = load_cfg(args.config_yaml)

    # conditions
    CLEAN = None
    ROTATION = {"rotation": {"enable": True, "random_k": True, "k_range": [-8, 8]},
                "permutation": {"enable": False}}
    PERMUTATION = {"rotation": {"enable": False},
                   "permutation": {"enable": True, "mode": "random"}}

    # model registry (label -> ckpt path)
    models = [
        ("baseline_noaug", args.baseline_noaug_ckpt),
        ("baseline_trainrot", args.baseline_rot_ckpt),
        ("set_noaug", args.set_noaug_ckpt),
        ("set_trainrot", args.set_rot_ckpt),
    ]
    models = [(n, p) for (n, p) in models if p]
    if not models:
        raise SystemExit("No checkpoints provided. Pass at least one of the --*_ckpt flags.")
    
    # Per-model YAML map (if provided)
    per_yaml = {
        "baseline_noaug": args.baseline_noaug_yaml,
        "baseline_trainrot": args.baseline_rot_yaml,
        "set_noaug": args.set_noaug_yaml,
        "set_trainrot": args.set_rot_yaml,
    }

    all_rows = []
    for model_name, ckpt in models:
        ckpt = os.path.expanduser(ckpt)
        if not os.path.exists(ckpt):
            print(f"[warn] Missing checkpoint {ckpt}; skipping {model_name}")
            continue

        # Load the right merged config for this model
        yaml_for_model = per_yaml.get(model_name)
        if yaml_for_model:
            cfg = load_cfg(yaml_for_model)
        else:
            cfg = global_cfg

        for cond_name, test_aug in [("clean", CLEAN), ("rotation", ROTATION), ("permutation", PERMUTATION)]:
            print(f"\n[info] Evaluating model={model_name} condition={cond_name} repeats={args.repeats}")
            rows = evaluate_model(ckpt_path=ckpt, base_cfg=cfg, test_aug=test_aug, repeats=args.repeats)
            # tag rows with model/condition
            for r in rows:
                r["model"] = model_name
                r["condition"] = cond_name
            all_rows.extend(rows)

            # Aggregate & print quick summary
            s = summarize(rows, metrics=("cer", "loss"))
            print(f"[summary] {model_name} | {cond_name} | CER: {s['cer']}, LOSS: {s['loss']}")

    # save per-run results
    out_csv = f"{args.out_prefix}.csv"
    out_json = f"{args.out_prefix}.json"
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    # aggregated summary
    groups = []
    for (model, condition), g in df.groupby(["model", "condition"]):
        s = summarize(g.to_dict("records"), metrics=("cer", "loss"))
        groups.append({
            "model": model,
            "condition": condition,
            "cer_mean": s["cer"]["mean"],
            "cer_std": s["cer"]["std"],
            "cer_min": s["cer"]["min"],
            "cer_max": s["cer"]["max"],
            "loss_mean": s["loss"]["mean"],
            "loss_std": s["loss"]["std"],
            "loss_min": s["loss"]["min"],
            "loss_max": s["loss"]["max"],
            "n": int(len(g)),
        })
    df_sum = pd.DataFrame(groups).sort_values(["model", "condition"])
    df_sum.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    print(f"\n[done] Wrote per-run metrics to: {Path(out_csv).resolve()}")
    print(f"[done] Wrote per-run JSON    to: {Path(out_json).resolve()}")
    print(f"[done] Wrote summary         to: {Path(args.out_prefix + '_summary.csv').resolve()}")


if __name__ == "__main__":
    # make matmul stable & deterministic-ish where possible
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(123, workers=True)
    main()
