# Simultaneous LoRA-SGD: factor-growth experiments

This project measures the squared spectral norms of the two LoRA factors during **ordinary simultaneous SGD**:

\[
S_t=\|A_t\|_2^2+\|B_t\|_2^2,
\qquad W_t=W_0+B_tA_t.
\]

The main experiment follows the image-classification framework in **Mu and Klabjan, “On the Convergence Rate of LoRA Gradient Descent”**, Section 4 and Appendix B: a low-rank logistic classifier on 512-dimensional ResNet-18 embeddings of CIFAR-10. An optional experiment adapts a frozen ResNet-18 on CIFAR-10 images. The optimizer here updates both factors from the **same pre-update iterate and the same minibatch**. It uses no momentum, clipping, rank scaling, Adam, or variance reduction.

The delivered code is a documented reconstruction for a new spectral-norm experiment, rather than a claim to reproduce the authors' figures exactly. Reported paper settings and additional implementation choices are separated in [the hyperparameter document](docs/HYPERPARAMETERS.md).

## What is included

| File | Purpose |
| --- | --- |
| `train_logistic.py` | NumPy implementation of simultaneous LoRA-SGD for multiclass cross-entropy on cached embeddings; also supports an offline synthetic smoke experiment |
| `extract_features.py` | Extract deterministic 512-dimensional CIFAR-10 embeddings using a frozen ResNet-18; requires PyTorch and Torchvision |
| `train_resnet.py` | Optional CIFAR-10 experiment with a frozen ResNet-18 backbone and trainable LoRA factors; requires PyTorch and Torchvision |
| `plot_results.py` | Plot factor norms from raw CSV files and compare runs |
| `lora_growth/metrics.py` | Numerically exact spectral-norm, Frobenius-norm, product, and imbalance diagnostics |
| `lora_growth/torch_layers.py` | Custom LoRA linear and convolutional layers for PyTorch |
| `lora_growth/torch_logger.py` | Reusable logger for custom LoRA and standard PEFT linear/convolutional adapters |
| `configs/` | Experiment configurations and/or example parameter records |
| `tests/` | Numerical checks for gradients, simultaneous updates, and matrix diagnostics |
| `docs/EXPERIMENT_PROTOCOL.md` | Objective, sampling, logging conventions, and interpretation of the results |
| `docs/HYPERPARAMETERS.md` | Paper settings, project choices, and recommended sweeps |
| `docs/INTEGRATION.md` | Add spectral-norm logging to an existing PyTorch/PEFT training loop |
| `docs/VALIDATION.md` | Executed checks and explicit scope of the bundled example results |

## 1. Install the lightweight environment

Use Python 3.10--3.12 when using the pinned vision dependencies. The lightweight NumPy path also works with newer Python versions supported by its dependencies. From the extracted project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead of the `source` command.

The NumPy classifier, synthetic smoke experiment, diagnostic functions, and plotting do not require a GPU or PyTorch. For the image feature extractor or ResNet experiment, also install a compatible PyTorch/Torchvision pair:

```bash
python -m pip install -r requirements-vision.txt
```

For a specific CUDA build, follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) instead. PyTorch and Torchvision must be mutually compatible. The original paper reports PyTorch 2.5.0 and CUDA 12.1; those are historical reproduction settings, not a statement that this project requires CUDA.

For CPU-only Linux or Windows, the compatible pinned pair can be installed explicitly before the vision requirements:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-vision.txt
```

For macOS, use the ordinary PyPI installation through `requirements-vision.txt`; do not use the Linux/Windows CPU wheel index above. The project pins PyTorch 2.5.1 and Torchvision 0.20.1, a paper-era compatible pair, rather than claiming to replicate the paper's exact patch version.

## 2. Run an offline smoke experiment

This checks the complete classifier-to-CSV-to-plot workflow without downloading data:

```bash
python train_logistic.py \
  --synthetic \
  --output results/synthetic_seed13 \
  --rank 4 \
  --epochs 3 \
  --batch-size 32 \
  --lr 0.05 \
  --seed 13 \
  --init-std 0.01 \
  --sampling iid \
  --log-every 1

python plot_results.py \
  --runs results/synthetic_seed13 \
  --output figures/synthetic_seed13
```

Use a new output directory for each run. Output directories are protected against accidental mixing or overwriting of runs. Synthetic data are an execution check and a controlled example; they are **not CIFAR-10 results**.

## 3. Prepare CIFAR-10 embeddings

The paper's logistic-regression experiment uses a ResNet-18 pretrained on ImageNet-1k as a frozen feature extractor. This project selects the standard Torchvision ImageNet weights and their deterministic preprocessing explicitly. The extraction script saves `x_train`, `y_train`, `x_test`, and `y_test` in an NPZ file, plus extraction metadata.

```bash
python extract_features.py \
  --output data/cifar10_resnet18_features.npz \
  --download
```

The first extraction needs internet access to download CIFAR-10 and the selected pretrained weights. Check `python extract_features.py --help` for dataset location, device, batch-size, and subset options. A subset extraction is useful to check a machine before the full run; use a different output file and label it as a subset experiment.

If you already have embeddings, supply an NPZ file with these arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `x_train` | `(N_train, d)` | Finite real-valued training features |
| `y_train` | `(N_train,)` | Integer training labels |
| `x_test` | `(N_test, d)` | Test features from the same extractor and preprocessing |
| `y_test` | `(N_test,)` | Integer test labels |

For the paper-inspired experiment, `d=512` and there are ten classes. Do not fit preprocessing on the test labels or silently mix feature caches from different checkpoints.

## 4. Train the simultaneous LoRA classifier

A baseline corresponding to the middle constant learning rate in the paper is:

```bash
python train_logistic.py \
  --features data/cifar10_resnet18_features.npz \
  --output results/cifar10_logistic_r4_lr005_seed13 \
  --rank 4 \
  --epochs 60 \
  --batch-size 512 \
  --lr 0.05 \
  --schedule constant \
  --sampling iid \
  --seed 13 \
  --init-std 0.01 \
  --lambda-product 0 \
  --log-every 1 \
  --eval-every 1
```

`iid` samples each minibatch with replacement, so the batch gradient is conditionally unbiased for the fixed empirical objective. The number of updates called one epoch is `ceil(N_train / batch_size)`. Use `--sampling shuffle` for a conventional shuffled pass through the dataset, and report that change: shuffled-without-replacement sampling is a different sampling scheme.

Both factor gradients are calculated before either factor is changed:

```python
grad_A = B.T @ grad_W
grad_B = grad_W @ A.T
A_new = A - learning_rate * grad_A
B_new = B - learning_rate * grad_B
A, B = A_new, B_new
```

The actual implementation uses the selected minibatch and the mean cross-entropy loss. Do not update `A` in place and then use that new value to calculate `grad_B`; that would be an alternating update.

### Learning-rate and seed sweeps

The paper's constant-rate logistic sweep is `0.03, 0.05, 0.07`. Run all three from the same initialization and use the same seed to couple the minibatch draws. For uncertainty across randomness, repeat each configuration with additional seeds:

```bash
for seed in 13 29 47; do
  for lr in 0.03 0.05 0.07; do
    python train_logistic.py \
      --features data/cifar10_resnet18_features.npz \
      --output "results/logistic_r4_lr${lr}_seed${seed}" \
      --rank 4 --epochs 60 --batch-size 512 \
      --lr "$lr" --seed "$seed" --init-std 0.01 \
      --schedule constant --sampling iid --log-every 1
  done
done
```

These loops are for Bash. The three seeds are project choices, not values reported in the paper.

The bundled sweep helper can launch the same kind of grid without a shell loop:

```bash
python sweep.py \
  --features data/cifar10_resnet18_features.npz \
  --output results/logistic_grid \
  --learning-rates 0.03 0.05 0.07 \
  --seeds 13 29 47 \
  --rank 4 --epochs 60 --batch-size 512
```

### Save parameters in JSON

The NumPy trainer also accepts `--config FILE`. The JSON object uses the argument names with underscores, such as `batch_size` and `init_std`. Explicit CLI options override the JSON defaults. For example, save the following as `my_run.json`:

```json
{
  "features": "data/cifar10_resnet18_features.npz",
  "output": "results/logistic_from_json",
  "rank": 4,
  "epochs": 60,
  "batch_size": 512,
  "lr": 0.05,
  "sampling": "iid",
  "seed": 13,
  "init_std": 0.01,
  "log_every": 1
}
```

Then run:

```bash
python train_logistic.py --config my_run.json
```

Unknown JSON keys are rejected. `--max-steps` caps the total update count for short checks; the saved horizon is adjusted accordingly. For synthetic runs, `--data-seed` controls the generated dataset independently of the initialization/sampling `--seed`; its default is `1729`.

### Product regularization

To examine a coercive sample-loss variant, set, for example:

```bash
python train_logistic.py \
  --features data/cifar10_resnet18_features.npz \
  --output results/logistic_regularized_seed13 \
  --rank 4 --epochs 60 --batch-size 512 --lr 0.05 \
  --seed 13 --init-std 0.01 --lambda-product 0.001
```

This adds `lambda_product * ||B @ A||_F^2 / 2` to the loss. It is a new experiment relative to the paper's unregularized image-classification setup. It is not the same as factor-wise optimizer weight decay.

### Conservative theorem-step diagnostic

`--schedule theory-general` selects the conservative fixed-horizon step cap documented by the program, using `--delta` for the failure-probability parameter. It is intended to check the factor-growth diagnostic under a theorem-motivated schedule; its learning rate can be much smaller than useful practical learning rates.

```bash
python train_logistic.py \
  --features data/cifar10_resnet18_features.npz \
  --output results/logistic_theory_seed13 \
  --rank 4 --epochs 60 --batch-size 512 \
  --schedule theory-general --delta 0.05 \
  --seed 13 --init-std 0.01 --log-every 1
```

Read the saved configuration for the actual step size and constants. A constant-rate run at `0.05` must not be described as satisfying the theorem's step-size restriction unless the recorded restriction is checked.

## 5. Plot the measured growth

```bash
python plot_results.py \
  --runs results/cifar10_logistic_r4_lr005_seed13 \
  --output figures/cifar10_logistic_r4_lr005_seed13
```

To compare runs, pass more run directories after `--runs`:

```bash
python plot_results.py \
  --runs results/logistic_r4_lr0.03_seed13 \
         results/logistic_r4_lr0.05_seed13 \
         results/logistic_r4_lr0.07_seed13 \
  --output figures/logistic_lr_comparison
```

The raw CSV files are the authoritative observations. Moving averages, logarithmic axes, or downsampling used only in plots do not change those files. A practical figure should show the three curves `a_spectral_sq`, `b_spectral_sq`, and `spectral_sum`, together with the loss or accuracy, so that apparent norm stabilization can be interpreted alongside optimization progress.

## 6. Optional ResNet-18 image experiment

```bash
python train_resnet.py \
  --output results/cifar10_resnet_r20_lr004_seed13 \
  --rank 20 \
  --epochs 250 \
  --batch-size 512 \
  --lr 0.04 \
  --seed 13 \
  --init-std 0.01 \
  --sampling iid \
  --log-every 1 \
  --download
```

Use `python train_resnet.py --help` for device, data, and short-run options. The frozen base is randomly initialized, and the LoRA factors are the trainable weights. The implementation adapts convolutional and linear layers; the paper specifically describes convolutional adaptation, so including the final linear layer is an explicit reconstruction choice. The program keeps the model in evaluation mode to freeze BatchNorm running statistics, following Appendix B. Evaluation mode does **not** disable gradient calculation; no training forward pass is enclosed in `torch.no_grad()`.

For a network containing multiple adapters, inspect the per-layer curves. The sum over layers and the maximum over layers are descriptive network-level summaries, not the single-pair quantity in the theorem. The ResNet experiment probes practical behavior; it does not establish global smoothness of the nonlinear network loss.

To execute the optional vision path without downloading CIFAR-10 or pretrained weights, use its explicitly labeled fake-data mode:

```bash
OMP_NUM_THREADS=2 python train_resnet.py \
  --output results/resnet_fake_smoke \
  --fake-data --max-train 16 --max-test 8 \
  --max-steps 2 --batch-size 4 --epochs 1 \
  --eval-limit 8 --device cpu

python extract_features.py \
  --output data/fake_features.npz \
  --fake-data --weights none \
  --max-train 16 --max-test 8 \
  --batch-size 4 --image-size 32 --device cpu
```

These are software execution checks only. They use generated images and must not be reported as CIFAR-10 validation. `OMP_NUM_THREADS=2` is Bash syntax that limits CPU threading for the tiny smoke run; it is not an optimization hyperparameter.

## 7. Understand the outputs

| Output | What it contains |
| --- | --- |
| `config.json` | Run arguments, initialization choices, and implementation/environment information |
| `pair_metrics.csv` | Raw per-pair factor norms and product/imbalance diagnostics at initialization and logging steps |
| `metrics.csv` | NumPy trainer: full-dataset evaluations and full empirical factor-gradient norm; ResNet trainer: per-update scalar loss/gradient measurements |
| `updates.csv` | NumPy classifier's loss, accuracy, and stochastic factor-gradient norm before each update |
| `aggregate.csv` | ResNet trainer's explicitly named sums/maxima across adapters |
| `evaluation.csv` | ResNet trainer's optional train/test loss and accuracy evaluations |
| `summary.json` | Final status and run-level diagnostic summaries |
| `initial_factors.npz` | Initial classifier factors and frozen base weights for exact reference |
| `final_factors.npz` | Final NumPy classifier factors and base weights |
| Model checkpoint from the ResNet trainer | Final adapter/model state and accompanying configuration |

`step=0` means before the first update and corresponds to `(A_1, B_1)` in the draft's one-based mathematical indexing. `step=k` is the state after `k` optimizer updates. The loss attached to an update may have been computed at its pre-update state; the norm record is explicitly the post-update state. Keep that distinction when aligning curves.

With `--log-every 1`, every finite iterate is measured. A larger interval reduces diagnostic overhead but gives a maximum over **logged** states only; it cannot certify that no larger value occurred between measurements. A non-finite loss, factor, gradient, or diagnostic should be treated as a failed run and retained in the run record, rather than omitted from a comparison.

## 8. Run the numerical checks

```bash
python -m unittest discover -s tests -v
```

The lightweight tests do not require CIFAR-10 or a GPU. PyTorch-specific checks are conditional on PyTorch being installed. See the accompanying validation record for which paths were actually executed when this bundle was prepared.

## Interpretation and limits

- `||A||_2` and `||B||_2` are matrix spectral norms, not Frobenius norms. The package logs both, under different column names.
- At initialization, `B=0`; a nonzero `A` is essential. Initializing both factors to zero would create a stationary factorization.
- A fixed-horizon bound `C_T = 2 S_0 + T^(1/4)` is a single horizontal reference for the run. It must not be replaced by the moving curve `2 S_0 + t^(1/4)` unless a separate result justifies that curve.
- A finite experiment can show observed growth, stabilization, or a bound violation for that run. It cannot prove boundedness for all time or a high-probability theorem.
- An upper bound of order `T^(1/4)` does not predict that the measured growth has exponent `1/4` or that the bound will be tight.
- Spectral factor growth depends on the parameterization. Rescaling `B` by `c` and `A` by `1/c` leaves the product unchanged while changing the individual norms.

Detailed reporting guidance is in [the experiment protocol](docs/EXPERIMENT_PROTOCOL.md).

## Included executed examples

`examples/` contains a nine-run **synthetic** logistic comparison (three learning
rates and three seeds), a small synthetic-image ResNet execution check, a cached
synthetic-feature pipeline check, and a conservative-step smoke run. The PNG/PDF
figures and raw CSV files are included. These examples demonstrate executable
code and logging; they are not CIFAR-10 experimental results. See
[VALIDATION.md](docs/VALIDATION.md) for the precise commands and test environment.

To instrument your own model without changing its training code, use
`FactorGrowthLogger` as shown in [INTEGRATION.md](docs/INTEGRATION.md). Log the
initialized state before the first optimizer update and log after each completed
joint update. Exact CPU SVD logging can be expensive for many convolutional
adapters; increasing `--log-every` changes the observed checkpoints and therefore
the scope of any reported maximum.

## References and implementation documentation

- Mu and Klabjan, *On the Convergence Rate of LoRA Gradient Descent*, Section 4 and Appendix B; author repository: <https://github.com/siqiaomu/lora>.
- PyTorch SGD: <https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html>.
- PyTorch singular values: <https://docs.pytorch.org/docs/stable/generated/torch.linalg.svdvals.html>.
- Torchvision ResNet-18 and pretrained transforms: <https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html>.

