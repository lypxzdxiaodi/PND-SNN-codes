# Anonymous PND implementation

This package contains the anonymous reference implementation and standalone
experiments for **Probabilistic Neuronal Dynamics (PND)**. It intentionally
contains no author names, repository links, server paths, checkpoints, datasets,
experiment logs, or reported benchmark numbers.

The primary implementation is `pnd/raw_snn_s4.py`. Historical class identifiers
are retained only for checkpoint and ablation compatibility. The main method
used by the current experiments is `psnns4`.

## 1. Method-to-code map

| Method quantity | Implementation | Shape |
| --- | --- | --- |
| expanded temporal modes | `MyKernel.C`, `log_A_real`, `A_imag`, `log_dt` | `(H,N/2,2)`, `(H,N/2)`, `(H,N/2)`, `(H,)` |
| effective modal coefficient | complex pole `-exp(log_A_real)+j*A_imag` | `(H,N/2)` |
| mode reduction | learned complex coefficient `C` and real conjugate-pair sum | `(H,N/2)` to `(H,L)` |
| mode response | `mechanism_analysis.common.mode_kernels` | `(H,N/2,L)` |
| compressed kernel | `MyKernel.forward` | `(H,L)` |
| membrane evidence | temporal convolution plus `D*u`, then `ReLU(aver*V)` | `(B,H,L)` |
| firing probability | `-expm1(-evidence)` | `(B,H,L)` |
| hard sample | binary hard Gumbel-Max | `(B,H,L)` |
| surrogate gradient | analytic probability straight-through path in `psnns4` | `(B,H,L)` |

`d_state=N` must be even because the implementation stores one component of
each complex-conjugate pair. A neuron therefore has `N/2` explicitly stored
temporal modes.

## 2. Core layer and ablations

| CLI name | Historical class | Purpose |
| --- | --- | --- |
| `pnd` | `psnns4` | main method: hard Gumbel-Max forward and probability-ST backward |
| `hard_gumbel` | `Snn_s4` | original hard-Gumbel relaxation |
| `bernoulli_st` | `bsnns4` | Bernoulli forward with probability-ST backward |
| `linear_gumbel` | `Snn_s4_linear` | learned two-logit firing without exponential mapping |
| `linear_bernoulli_st` | `bsnns4_linear` | linear logits with Bernoulli probability-ST |
| `bidirectional_pnd` | `bidir_psnns4` | exploratory non-causal temporal response |
| `sigmoid_pnd` | `SigmoidPND` | sigmoid probability comparison |
| `--only-spike` | model-level ablation | removes post-spike dropout/projection from each residual branch |

The compatibility file also retains the historical `pif` and binary-relaxation
helpers. They are not used by the main method.

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The package does not download data automatically unless `--download` is passed
for CIFAR-10. LRA data should be obtained from the official Long Range Arena
release and supplied with `--data-dir`.

## 4. Standalone training

All tasks use validation accuracy for checkpoint selection. The test split is
evaluated once after loading `best.pt`. Every output directory contains:

```text
config.json
history.jsonl
best.pt
metrics.json
```

### Sequential grayscale CIFAR-10

The image is converted to grayscale and flattened to `(B,1024,1)`.

```bash
python -m experiments.train \
  --task scifar --data-dir ./data/cifar --output runs/scifar \
  --layer pnd --epochs 400 --d-model 256 --d-state 128
```

For offline/reviewer environments, leave `--download` absent as shown. Add it
only when an explicit download is desired.

### ListOps

```bash
python -m experiments.train \
  --task listops --data-dir ./data/lra/listops-1000 \
  --output runs/listops --layer pnd --epochs 200
```

### IMDB character classification

```bash
python -m experiments.train \
  --task imdb --data-dir ./data/hf_cache \
  --output runs/imdb --layer pnd --epochs 200
```

### AAN paired character sequences

```bash
python -m experiments.train \
  --task aan --data-dir ./data/lra/tsv_data \
  --output runs/aan --layer pnd --epochs 200
```

### Pathfinder-32

```bash
python -m experiments.train \
  --task pathfinder --data-dir ./data/lra/pathfinder32 \
  --output runs/pathfinder --layer pnd --epochs 200
```

The loader uses `curv_contour_length_14`, deterministic 80/10/10 splits, and a
flattened `(B,1024,1)` input. Add `--center` only for the centered-input
ablation.

### MNIST, N-MNIST, DVS128 Gesture, and Speech Commands

These optional modalities use a separate standalone entry:

```bash
pip install -r requirements-optional.txt
python -m experiments.train_modalities --task mnist --data-dir ./data/mnist --output runs/mnist
python -m experiments.train_modalities --task nmnist --data-dir ./data/nmnist --output runs/nmnist --frames 20
python -m experiments.train_modalities --task dvs128 --data-dir ./data/dvs128 --output runs/dvs128 --frames 50 --batch-size 8
python -m experiments.train_modalities --task speechcommands --data-dir ./data/speech_commands --output runs/speechcommands
```

The event-camera entry uses a learned and checkpointed spatial patch projection.
This replaces the historical N-MNIST preprocessing experiment that created an
untrained random linear transform inside the dataset transform. The release
implementation is deterministic, trainable, and fully represented in the
checkpoint. Speech Commands labels use a sorted directory list so class indices
do not depend on filesystem traversal order.

## 5. Mechanism experiments

Every analysis writes `config.json`, `metrics.json`, `raw_data.npz`, and a
`figures/` directory when a figure is applicable. Figures can therefore be
regenerated from saved raw arrays.

### RQ1/RQ2: temporal diversity and redundancy

```bash
python -m mechanism_analysis.analyze_temporal_basis \
  --checkpoint runs/scifar/best.pt --output outputs/temporal_basis \
  --length 1024 --channels 6
```

This script jointly computes:

- individual modal response curves and their compressed kernel;
- variance of local log-magnitude slopes, measuring departure from one fixed
  geometric response;
- pairwise absolute mode correlation, measuring temporal-basis redundancy;
- normalized singular-value spectrum and entropy effective rank.

Run the same command on a saved initialization checkpoint to obtain the
initialization-versus-trained comparison. Random and high-amplitude channels are
both selected; the selection labels are stored in `metrics.json`.

### RQ3: time-resolved task evidence

```bash
python -m mechanism_analysis.analyze_linear_probe \
  --checkpoint runs/scifar/best.pt --data-dir ./data/cifar \
  --output outputs/linear_probe
```

The backbone is frozen. Independent linear classifiers are trained at selected
sequence fractions to measure when class information becomes linearly readable.

### RQ4: temperature sensitivity

```bash
python -m mechanism_analysis.analyze_temperature \
  --checkpoint runs/scifar/best.pt --data-dir ./data/cifar \
  --output outputs/temperature --repeats 10
```

The default grid is `0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0`. For the main
probability-ST estimator, temperature does not enter the analytic backward path,
and a positive common temperature does not change the hard categorical argmax.
This analysis verifies that implementation-level invariance instead of assuming
that an accuracy difference must exist.

### RQ5: repeated stochastic inference

```bash
python -m mechanism_analysis.analyze_stochastic_inference \
  --checkpoint runs/scifar/best.pt --data-dir ./data/cifar \
  --output outputs/stochastic_inference --repeats 10
```

It reports accuracy mean/std/min/max, per-sample majority agreement, full
agreement, at-least-90% agreement, and unstable-sample fractions. Only the
inference RNG changes.

### Why exponential probability?

Train matched models:

```bash
python -m experiments.train --task scifar --data-dir ./data/cifar \
  --output runs/probability/exponential --layer pnd --epochs 400
python -m experiments.train --task scifar --data-dir ./data/cifar \
  --output runs/probability/sigmoid --layer sigmoid_pnd --epochs 400
```

Then compare probability distributions, accuracy, confidence, and layer-wise
hard firing rates:

```bash
python -m mechanism_analysis.analyze_probability_comparison \
  --exponential-checkpoint runs/probability/exponential/best.pt \
  --sigmoid-checkpoint runs/probability/sigmoid/best.pt \
  --data-dir ./data/cifar --output outputs/probability_comparison
```

Convergence is read from the two `history.jsonl` files. Local gradient variance
and stability are measured separately:

```bash
python -m mechanism_analysis.analyze_gradient_estimators \
  --output outputs/gradient_estimators
```

### Linear-expansion dimension

```bash
python -m experiments.run_dstate_sweep \
  --task scifar --data-dir ./data/cifar --output runs/d_state \
  --states 2 4 8 16 32 64 128 --epochs 400
```

This is a task-performance ablation. It should be interpreted jointly with the
effective-rank analysis; a larger configured expansion is not evidence that all
modes are used.

### Pure spike output

```bash
python -m experiments.train \
  --task scifar --data-dir ./data/cifar --output runs/only_spike \
  --layer pnd --only-spike --epochs 400
```

This removes the post-spike channel projection from the residual branch while
leaving temporal dynamics, probability estimation, hard sampling, residual
connections, pooling, and classifier unchanged.

## 6. Reproducibility scope

- Seeds cover Python, NumPy, PyTorch, CUDA, split generators, and DataLoader
  workers.
- Checkpoints are selected only with the validation split.
- The test split is evaluated only after training.
- Raw analysis values are saved independently of plots.
- No benchmark result in this package should be inferred from filenames or
  defaults; reported values must come from the generated metrics files.
