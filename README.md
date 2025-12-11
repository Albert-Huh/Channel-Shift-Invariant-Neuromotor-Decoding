# Channel-Shift-Invariant Neuromotor Decoding (Handwriting)

> A permutation/rotation-robust decoder built **on top of** Meta’s Generic Neuromotor Interface ([ [`Paper`](https://www.nature.com/articles/s41586-025-09255-w) ] [ [`BibTeX`](#citation) ]) for the handwriting task. This repo adds a **Set‑Transformer** backbone and **channel‑space data augmentation** to improve robustness to don/doff (band on/off), band rotation, and electrode misplacement.

---

## TL;DR
- **Problem:** EMG channel layout changes across wear sessions → baseline decoders are sensitive to channel order/rotation.
- **Idea:** Treat channels as an **unordered set** (Set‑Transformer encoder) + **channel‑space augmentation** (rotation, permutation) to learn spatially robust representations.
- **Scope:** Handwriting task on the public GNI dataset (80 train / 10 val / 10 test users). Baselines kept intact for fair comparison.

---

## What’s new in this repo (vs. upstream GNI)
1. **Set‑Transformer backbone** for handwriting
   - Toggle via config: `lightning_module.network.use_set_transformer={true|false}`.
   - Drop‑in replacement; downstream decoder (CTC, Conformer) and metrics unchanged.
2. **Channel‑space augmentation** built into the DataModule (per split)
   - Hydra keys under `data_module.channel_aug.{train|val|test}`.
   - Augs supported: **circular rotation** (random/fixed), **random permutation** (stress test).
3. **Robustness evaluation utilities**
   - `eval_results.py` – run **clean**, **rotation**, and **permutation** test conditions **N** repeats per model; aggregates mean/std/min/max.
4. **Label export utilities** (optional, for analysis)
   - `export_handwriting_labels_all_splits.py` – dumps ground‑truth labels (train/val/test) to JSON/TSV.
5. **Notebooks & scripts** for quick starts
   - Based on upstream `handwriting-eval.ipynb`, plus training/eval automation and augmentation toggles.

---

## Repository structure (key files)
```
./
├─ run.sh                         # Minimal, block‑style commands for baseline / SetTx + aug variants
├─ eval_results.py                # Evaluate checkpoints under clean/rotation/permutation (repeat N times)
├─ export_handwriting_labels_all_splits.py   # (optional) Export reference labels
├─ notebooks/
│  └─ handwriting-eval.ipynb      # Example inference/eval (clean + augmented)
├─ generic_neuromotor_interface/
│  ├─ networks.py                  # Handwriting nets; Set‑Transformer toggle via use_set_transformer
│  ├─ data_module.py               # DataModule with channel_aug per split (train/val/test)
│  └─ ...                          # (upstream GNI modules retained)
└─ ...
```

---

## Setup

### 1) Environment
```bash
# Python >=3.10 recommended
conda create -n neuromotor python=3.10 -y
conda activate neuromotor
pip install -e .
```

### 2) Data & (optional) pretrained weights
```bash
# Download the public handwriting subset or full data to ./data
python -m generic_neuromotor_interface.scripts.download_data handwriting full_data ./data
# (optional) Pretrained model
python -m generic_neuromotor_interface.scripts.download_models handwriting ./models
```

> The dataset CSV is expected at `./data/handwriting_corpus.csv`.

---

## Training
**We keep validation clean** (no aug) to ensure early‑stopping & best checkpoint selection are comparable across runs. Edit `run.sh` and uncomment **one block** at a time.

### Baseline (no augmentation)
```bash
python -m generic_neuromotor_interface.train \
  --config-name=handwriting \
  data_location=$(pwd)/data \
  trainer.max_epochs=40 \
  lightning_module.network.use_set_transformer=false \
  +callbacks.1.save_top_k=-1 +callbacks.1.every_n_epochs=1 \
  +data_module.channel_aug.train.rotation.enable=false \
  +data_module.channel_aug.train.permutation.enable=false \
  +data_module.channel_aug.val.rotation.enable=false \
  +data_module.channel_aug.val.permutation.enable=false \
  +data_module.channel_aug.test.rotation.enable=false \
  +data_module.channel_aug.test.permutation.enable=false
```

### Baseline (train‑time rotation; val/test clean)
```bash
python -m generic_neuromotor_interface.train \
  --config-name=handwriting \
  data_location=$(pwd)/data \
  trainer.max_epochs=40 \
  lightning_module.network.use_set_transformer=false \
  +callbacks.1.save_top_k=-1 +callbacks.1.every_n_epochs=1 \
  +data_module.channel_aug.train.rotation.enable=true \
  +data_module.channel_aug.train.rotation.random_k=true \
  +data_module.channel_aug.train.rotation.k_range=[-8,8] \
  +data_module.channel_aug.train.permutation.enable=false \
  +data_module.channel_aug.val.rotation.enable=false \
  +data_module.channel_aug.val.permutation.enable=false \
  +data_module.channel_aug.test.rotation.enable=false \
  +data_module.channel_aug.test.permutation.enable=false
```

### Set‑Transformer (no augmentation)
```bash
python -m generic_neuromotor_interface.train \
  --config-name=handwriting \
  data_location=$(pwd)/data \
  trainer.max_epochs=40 \
  lightning_module.network.use_set_transformer=true \
  +callbacks.1.save_top_k=-1 +callbacks.1.every_n_epochs=1 \
  +data_module.channel_aug.train.rotation.enable=false \
  +data_module.channel_aug.train.permutation.enable=false \
  +data_module.channel_aug.val.rotation.enable=false \
  +data_module.channel_aug.val.permutation.enable=false \
  +data_module.channel_aug.test.rotation.enable=false \
  +data_module.channel_aug.test.permutation.enable=false
```

### Set‑Transformer (train‑time rotation; val/test clean)
```bash
python -m generic_neuromotor_interface.train \
  --config-name=handwriting \
  data_location=$(pwd)/data \
  trainer.max_epochs=40 \
  lightning_module.network.use_set_transformer=true \
  +callbacks.1.save_top_k=-1 +callbacks.1.every_n_epochs=1 \
  +data_module.channel_aug.train.rotation.enable=true \
  +data_module.channel_aug.train.rotation.random_k=true \
  +data_module.channel_aug.train.rotation.k_range=[-8,8] \
  +data_module.channel_aug.train.permutation.enable=false \
  +data_module.channel_aug.val.rotation.enable=false \
  +data_module.channel_aug.val.permutation.enable=false \
  +data_module.channel_aug.test.rotation.enable=false \
  +data_module.channel_aug.test.permutation.enable=false
```

> **Fixed rotation** instead of random: set `random_k=false` and `k=<int>` (e.g., `k=4`).

---

## Evaluation

### Quick, single‑pass tests
**Clean test:**
```bash
python -m generic_neuromotor_interface.train \
  --config-name=handwriting \
  train=false eval=false test=true \
  ckpt_path=/abs/path/to/best.ckpt \
  data_location=$(pwd)/data \
  +data_module.channel_aug.test.rotation.enable=false \
  +data_module.channel_aug.test.permutation.enable=false
```

**Rotation stress (random k in [-8,8]):**
```bash
python -m generic_neuromotor_interface.train \
  --config-name=handwriting \
  train=false eval=false test=true \
  ckpt_path=/abs/path/to/best.ckpt \
  data_location=$(pwd)/data \
  +data_module.channel_aug.test.rotation.enable=true \
  +data_module.channel_aug.test.rotation.random_k=true \
  +data_module.channel_aug.test.rotation.k_range=[-8,8] \
  +data_module.channel_aug.test.permutation.enable=false
```

**Permutation stress (rotation disabled):**
```bash
python -m generic_neuromotor_interface.train \
  --config-name=handwriting \
  train=false eval=false test=true \
  ckpt_path=/abs/path/to/best.ckpt \
  data_location=$(pwd)/data \
  +data_module.channel_aug.test.rotation.enable=false \
  +data_module.channel_aug.test.permutation.enable=true \
  +data_module.channel_aug.test.permutation.mode=random
```

### Robustness with repeats (aggregated stats)
Use `eval_results.py` to run **N repeats** per condition and aggregate metrics:
```bash
python eval_results.py \
  --data_dir $(pwd)/data \
  --config_yaml /path/to/merged_config.yaml \  # per‑model merged Hydra config
  --repeats 10 \
  --baseline_noaug_ckpt /runs/baseline_noaug/best.ckpt \
  --baseline_rot_ckpt   /runs/baseline_trainrot/best.ckpt \
  --set_noaug_ckpt      /runs/set_noaug/best.ckpt \
  --set_rot_ckpt        /runs/set_trainrot/best.ckpt \
  --baseline_noaug_yaml /runs/baseline_noaug/.hydra/config.yaml \
  --baseline_rot_yaml   /runs/baseline_trainrot/.hydra/config.yaml \
  --set_noaug_yaml      /runs/set_noaug/.hydra/config.yaml \
  --set_rot_yaml        /runs/set_trainrot/.hydra/config.yaml \
  --out_prefix results/handwriting
```
Outputs:
- `results/handwriting.csv` – per‑run metrics
- `results/handwriting_summary.csv` – mean/std/min/max per (model × condition)

---

## Configuration reference (augmentation)
All augmentation knobs live under `data_module.channel_aug` and can be set **per split**:
```yaml
# example
data_module:
  channel_aug:
    train:
      rotation:    { enable: true,  random_k: true, k_range: [-8, 8] }  # or: { enable: true, random_k: false, k: 4 }
      permutation: { enable: false }
    val:
      rotation:    { enable: false }
      permutation: { enable: false }
    test:
      rotation:    { enable: true,  random_k: true, k_range: [-8, 8] }
      permutation: { enable: false }
```

---

## Limitations & future work
- Permutation‑only stress can be harsher than real don/doff; include cross‑session analyses when possible.
- Channel dropout augmentation is prototyped but disabled by default; consider enabling for missing‑sensor scenarios.
- Language‑model post‑processing (spell/grammar) can further reduce effective CER without touching the decoder.

---

## Acknowledgements
- Built on **Meta’s Generic Neuromotor Interface** (GNI) for the handwriting task. Please see the upstream license and citations in the original repository.

## License
This project inherits the upstream licensing terms where applicable. See `LICENSE` for details.

## License

The dataset and the code are CC-BY-NC-4.0 licensed, as found in the LICENSE file.


