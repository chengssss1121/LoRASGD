# Executed validation record

This record describes checks actually executed while preparing the code bundle.
It does not describe completed CIFAR-10 training.

## Environment

- Python 3.12.14 on Linux, CPU execution.
- NumPy classifier/plot path: NumPy 2.3.5, Matplotlib 3.10.8.
- Optional vision and logger tests: PyTorch 2.5.1+cpu, Torchvision 0.20.1+cpu;
  their isolated dependency path used NumPy 2.5.4.
- No GPU run was performed. CUDA/MPS execution may have different performance
  and numerical reproducibility characteristics.

## Numerical and integration checks

All **25 tests passed** with optional PyTorch installed:

```bash
python -m unittest discover -s tests -v
```

The checks include central finite-difference verification of factor gradients,
single-step simultaneous updates using both old factors, zero-B initialization,
frozen base parameters, convolution geometry, frozen BatchNorm statistics,
spectral-versus-Frobenius distinctions, reduced-QR product norms, parameterization
rescaling, full-gradient chunking, and invariance of training to logging cadence.
Reusable-logger checks additionally cover multiple PEFT-style adapter dictionaries,
adapter scaling, and unchanged parameters, gradients, module modes, and RNG state.
PEFT discovery was tested with modules implementing its adapter interface; an
actual downloaded Hugging Face model was not loaded.

The included logistic examples contain 8,649 norm records (nine runs, each with
960 updates plus initialization). Each final logged spectral metric was checked
against a fresh SVD of the saved final factors. Initial B norms are zero, logged
update indices are complete, and maxima agree with the raw records.

## Executed synthetic logistic sweep

```bash
python sweep.py --synthetic \
  --output examples/logistic_synthetic_sweep \
  --learning-rates 0.03 0.05 0.07 --seeds 13 23 37 \
  --epochs 60 --batch-size 32 --rank 4
```

These use a common generated dataset (512 training examples, 256 test examples,
32 features, 10 classes; data seed 1729). Batch size 32 is an execution-example
choice, not the paper's batch size 512. All nine runs completed with finite
factor measurements. The recorded loss and factor trajectories are observations
for this generated dataset only, not a convergence-rate or CIFAR-10 claim.

Plots are included in `examples/figures_logistic_synthetic/`. Curves show
means over three seeds at each learning rate; bands are plus/minus one sample
standard deviation, not confidence intervals. Individual raw runs are retained.

## Executed optional vision paths

The following offline checks used generated images:

```bash
python extract_features.py --fake-data --weights none \
  --output examples/fake_resnet18_features.npz \
  --max-train 16 --max-test 8 --batch-size 4 \
  --image-size 32 --device cpu

python train_resnet.py --fake-data \
  --output examples/resnet_fake_smoke \
  --max-train 16 --max-test 8 --max-steps 2 \
  --batch-size 4 --epochs 1 --eval-limit 8 --device cpu
```

Feature extraction produced arrays of shape `(16,512)` and `(8,512)`. The ResNet
check performed two joint updates and logged 21 factor pairs at steps 0, 1, and
2, for 63 per-pair rows. All initial B norms were zero and final measurements
were finite. Its full model checkpoint is excluded from the compact archive;
the training script creates a full checkpoint on actual runs.

The cached-feature NumPy path was also exercised:

```bash
python train_logistic.py \
  --features examples/fake_resnet18_features.npz \
  --output examples/cached_feature_pipeline_smoke \
  --rank 4 --epochs 2 --batch-size 4 --lr 0.03 \
  --sampling shuffle --lambda-product 0.001
```

## Executed theorem-step diagnostic

```bash
python train_logistic.py --synthetic \
  --output examples/theory_step_smoke \
  --rank 4 --epochs 2 --batch-size 32 \
  --schedule theory-general --max-steps 16 --delta 0.05
```

The generated run used the recorded conservative step and fixed terminal
horizon. No violation of its reference C_T occurred in this short run. This
single observation does not establish a probability guarantee.

## What remains to run for the empirical paper

Download CIFAR-10 and the selected ImageNet checkpoint, generate the full feature
cache, and run the documented 60-epoch classifier grid and optional 250-epoch
ResNet grid. The full CIFAR-10 download was not completed in this preparation
session. No pretrained-checkpoint extraction, full CIFAR-10 training, GPU
benchmark, or reproduction of the authors' numerical results is claimed.

Use fresh output paths when repeating commands because existing runs are
protected against accidental overwrite.
