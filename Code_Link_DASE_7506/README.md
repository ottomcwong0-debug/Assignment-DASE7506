## 1. Install

Use **Python 3.12**. From the extracted package directory:

```bash
cd Code_Link_DASE_7506
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch for **one** device:

```bash
# Linux/Windows CPU: recommended; no GPU needed
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this command **instead**:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

For macOS, install `torch==2.7.1` from the default PyPI index and run on CPU. After installing PyTorch, install the remaining dependencies and check the model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux CPU commands were verified with Python 3.12 and PyTorch 2.7.1+cpu. Windows/macOS timings have not been measured.

## 2. Train and evaluate

**Quick installation check** — 10 training steps, then full-test evaluation:

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

This checks that the pipeline works; its score is **not** the full baseline. Each training run needs a new output directory.

**Full baseline** — 1,200 training updates, then evaluation:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

The baseline has four GPT blocks, width 128, four attention heads and **1,088,256 parameters**, and achieves approximately **2.10 test BPB**. On the reference four-thread Xeon Platinum 8457C, measured training took about **311 seconds** and scoring **5.92 seconds**, excluding installation and loading. These are reference measurements, not laptop guarantees or a fixed time allowance.

**My model** — :

```bash
python train.py --implementation student --seed 17 --eval-every 300 --steps 5000 --threads 8 --run-dir runs/swiglu-max
python evaluate.py --checkpoint runs/swiglu-max/checkpoint.pt --split validation
# Freeze the final method before testing:
python evaluate.py --checkpoint runs/swiglu-max/checkpoint.pt --split test
```

Training writes `checkpoint.pt` and `metrics.json`. Evaluation writes `test_cpu_fp32.json` (or the corresponding device/split name) and per-window losses. Submit the **bpb** value from the complete-test JSON, not token perplexity or validation BPB. Default evaluation is FP32. Add `--device cuda` for GPU runs; training can use BF16, but ranked evaluation must use FP32 and remain reproducible on CPU. The supplied CUDA runner caps PyTorch allocation at 20 GB; driver overhead is additional.

## 3. Files and model interface

| Files | Use |
|---|---|
| `model.py`, `configs/baseline.json` | Runnable baseline; preserve for comparisons. |
| `student.py`, `train.py` | My model factory and training recipe; added supporting code as needed. |
| `common.py`, `evaluate.py` | Fixed data checks, windows and scorer; keep unchanged. |
| `data/` | Supplied splits, tokenizer and dataset hashes; keep unchanged. |
| `tests/test_contract.py` | Checks your model's causality, normalization, independence and gradients. |
| `PACKAGE_MANIFEST.json` | Release hashes; paths are relative to the package root containing code/ and guide/. |

- `build_model(config)` returns a PyTorch model with `context=256`.
- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have shape `[batch, time, 2048]`.
- A prediction at position t may use only the observed prefix through t. Reset temporary state between independent windows, examples and scoring passes. Compact training-derived assets may be reused across windows; evaluation-prefix state may not.
- Checkpoints record the implementation module and configuration. Include that module and every required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is required for direct evaluation.
- Training length, architecture, optimizer, regularization, self-trained weight averaging and ensembles may change within the guide's constraints. Log all seeds, processed training targets, checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No particular seed or score improvement is mandated.

## 4. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048, independent windows of 256 targets, including the final short window. Every target except the first token of each split is scored once. Input windows share a boundary token but carry no state. BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8 byte length, including the first token's bytes.

| Split | Scored targets | UTF-8 bytes |
|---|---:|---:|
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

Use validation for all development and checkpoint/mixture selection. Weights, statistics and retrieval entries must derive only from training text. The public test text enables reproduction; it must not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for timing or reproduction. Token perplexity is not directly comparable with published word-level perplexity.

Measure all three limits for the same frozen predictor:

- **CPU time ≤5× baseline:**
- **Peak RAM ≤4 GiB:**
- **Inference assets ≤64 MiB uncompressed:** 


