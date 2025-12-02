#!/usr/bin/env python3
"""
Export all handwriting labels (train/val/test) to JSON and TSV.

- Reads <data_dir>/handwriting_corpus.csv to discover datasets & splits
- Builds each dataset using transforms/padding from model_config.yaml
- Instantiates the decoder from the config to get the charset (no checkpoint needed)
- Writes a single JSON (with "split" per record) and a TSV

Usage:
  python export_handwriting_labels_all_splits.py \
    --data_dir ~/emg_data \
    --config_yaml ~/emg_models/handwriting/model_config.yaml \
    --sort_names
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

# repo imports
from generic_neuromotor_interface.data import make_handwriting_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Folder containing handwriting_corpus.csv and HDF5 files")
    p.add_argument("--config_yaml", required=True,
                   help="Path to model_config.yaml (for transform/padding/decoder)")
    p.add_argument("--splits", default="train,val,test",
                   help="Comma-separated splits to export (choices: train,val/test). Default: all")
    p.add_argument("--json_out", default="handwriting_labels_all.json",
                   help="Combined JSON output path")
    p.add_argument("--tsv_out", default="handwriting_labels_all.tsv",
                   help="Combined TSV output path")
    p.add_argument("--sort_names", action="store_true",
                   help="Sort dataset names within each split for deterministic order")
    p.add_argument("--max_datasets_per_split", type=int, default=None,
                   help="(Optional) limit number of datasets per split for smoke-tests")
    return p.parse_args()


def get_charset_from_config(cfg):
    try_paths = []
    if "lightning_module" in cfg and "decoder" in cfg.lightning_module:
        try_paths.append(cfg.lightning_module.decoder)
    if "decoder" in cfg:
        try_paths.append(cfg.decoder)

    for node in try_paths:
        try:
            decoder = instantiate(node)
            if hasattr(decoder, "_charset") and hasattr(decoder._charset, "labels_to_str"):
                return decoder._charset
        except Exception:
            pass
    return None


def main():
    args = parse_args()
    data_dir = os.path.expanduser(args.data_dir)

    cfg = OmegaConf.load(os.path.expanduser(args.config_yaml))
    cfg.data_module.data_location = data_dir
    if "from_csv" in cfg.data_module.data_split._target_:
        cfg.data_module.data_split.csv_filename = os.path.join(data_dir, "handwriting_corpus.csv")
    charset = get_charset_from_config(cfg)
    if charset is None:
        print("[warn] Could not instantiate decoder from config; will export label_ids only (no text).")
    corpus_csv = Path(data_dir) / "handwriting_corpus.csv"
    if not corpus_csv.exists():
        raise FileNotFoundError(f"Missing corpus CSV: {corpus_csv}")
    df = pd.read_csv(corpus_csv)
    dataset_col = "dataset_name" if "dataset_name" in df.columns else ("dataset" if "dataset" in df.columns else None)
    split_col = "split" if "split" in df.columns else None
    if dataset_col is None or split_col is None:
        raise RuntimeError(f"Unexpected corpus schema in {corpus_csv}; need 'split' and 'dataset_name' (or 'dataset').")

    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    valid = {"train", "val", "eval", "test"}

    by_split = {}
    for s in requested_splits:
        q = df[df[split_col] == s]
        names = q[dataset_col].unique().tolist()
        if args.sort_names:
            names = sorted(names)
        if args.max_datasets_per_split is not None:
            names = names[: args.max_datasets_per_split]
        by_split[s] = names
        print(f"[info] Split '{s}': {len(names)} datasets")

    transform = instantiate(cfg.data_module.transform)
    padding = tuple(cfg.data_module.padding)
    min_duration_s = float(getattr(cfg.data_module, "min_duration_s", 0.0))
    records = []
    total_prompts = 0

    for split_name, dataset_names in by_split.items():
        for dname in dataset_names:
            ds = make_handwriting_dataset(
                dataset_names=[dname],
                data_location=cfg.data_module.data_location,
                transform=transform,
                padding=padding,
                emg_augmentation=None,
                concatenate_prompts=False,
                min_duration_s=min_duration_s,
            )

            # parse user/dataset from e.g., handwriting_user_001_dataset_000
            user_num = dataset_num = None
            m = re.search(r"user_(\d+)_dataset_(\d+)", dname)
            if m:
                user_num = int(m.group(1))
                dataset_num = int(m.group(2))

            for i in range(len(ds)):
                sample = ds[i]
                label_ids = sample["prompts"]
                label_text = charset.labels_to_str(label_ids) if charset is not None else None

                rec = {
                    "split": split_name,
                    "dataset_name": dname,
                    "prompt_index": int(i),
                    "label": label_text,
                    "label_ids": [int(x) for x in label_ids.tolist()],
                }
                if user_num is not None:
                    rec["user_number"] = user_num
                if dataset_num is not None:
                    rec["dataset_number"] = dataset_num
                if "timestamps" in sample:
                    rec["timestamps_len"] = int(len(sample["timestamps"]))
                records.append(rec)
                total_prompts += 1

            print(f"[info] {split_name} | {dname}: {len(ds)} prompts")

    # save outputs
    json_out = Path(args.json_out)
    tsv_out = Path(args.tsv_out)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with tsv_out.open("w", encoding="utf-8") as f:
        # TSV: split \t dataset_name \t prompt_index \t label
        for r in records:
            f.write(f"{r['split']}\t{r['dataset_name']}\t{r['prompt_index']}\t{(r['label'] or '')}\n")

    print(f"[done] Exported {total_prompts} prompts from {sum(len(v) for v in by_split.values())} datasets")
    print(f"JSON: {json_out.resolve()}")
    print(f"TSV : {tsv_out.resolve()}")


if __name__ == "__main__":
    # it is deterministic enough for ordering and no effect on labels themselves
    torch.manual_seed(42)
    main()