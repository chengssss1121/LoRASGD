# ViT LoRA-SGD: factor growth and gradient-noise amplification

This experiment uses an ordinary constant learning rate. It does not use horizon calibration or tune a learning rate on the test set. It complements the CIFAR-10/ResNet-18 experiment by measuring a mechanism in the theory on CIFAR-100 and a pretrained ViT.

## Scientific question

Does the evolution of the low-rank factors materially change the noise seen by the factor updates, even when the training loss decreases? How closely does $\|A_t\|_2^2+\|B_t\|_2^2$ describe the ratio between factor-gradient and weight-gradient noise variances?

The reference *On the Convergence of LoRA under Generalized Smoothness*, Section 6.3/Figure 3 and Appendix G.3, measures weight-gradient and factor-gradient noise at the same frozen iterates. Section 6.4/Figure 4 and Appendix G.4 extend this diagnostic to ViT using a fixed learning rate. We adopt this experimental question, while using the spectral factor-size quantity from your analysis. The reference's learning rate 0.3 is not transferred to this different adaptation setup.

## Model and defaults

- Backbone: `google/vit-base-patch16-224-in21k`, evaluation mode, CIFAR-100.
- One rank-4 adapter in the final transformer block's MLP output projection. The pretrained projection, all other backbone parameters, and the classifier after pilot warmup are frozen.
- No adapter scaling multiplier, momentum, weight decay, or clipping.
- Initialization: independent entries of A have distribution N(0, 0.01^2), and B is zero.
- Training: 4,000 simultaneous SGD updates, training batch size 64, IID sampling with replacement, seeds 13, 14, 15, 16, 17.
- A fixed 5,000-example pilot partition is disjoint from the 45,000-example main training partition. The pilot uses 4,000 fit examples and 1,000 validation examples. The head is warmed up on pilot fit only, then frozen.
- Select a single constant learning rate from 0.003, 0.01, 0.03 using pilot validation loss after 200 pilot updates. The same selected rate and frozen head are used for all five main runs. Alternatively, `--lr 0.01` specifies the rate directly and skips rate selection. Pilot head warmup still runs.
- Diagnostics: initial state and 20 evenly spaced intervals, including the final state. At each frozen checkpoint, diagnostic batch sizes are 16, 64, 256, with 32 independent pairs per batch size. These sizes change only noise measurement, not the training batches or trajectory.

The cached model computes `head(LayerNorm(r0 + (u @ A.T) @ B.T))`, with final LayerNorm parameters fixed. Here u is the pretrained final MLP input at the CLS token, and r0 is the pretrained CLS representation before the final LayerNorm. Because there is no later token mixing, this reproduces an adapter in that projection for the CLS classifier. Preparation checks the cached path against the full model, including a nonzero rank-one perturbation. This is internal single-layer ViT adaptation, not adaptation of every attention projection.

## What is measured

Use the manuscript's notation $\xi_t=g_t-G_t$. In this minibatch experiment, $g_t$ denotes the gradient averaged over a diagnostic minibatch, and $G_t$ denotes the full empirical weight gradient at the same iterate. Expectations below are over diagnostic minibatches with the training set and $(B_t,A_t)$ fixed.

The two noise panels estimate

$$
\mathbb E\|\xi_t\|_F^2
\quad\text{and}\quad
\mathbb E\bigl[\|B_t^\top\xi_t\|_F^2+\|\xi_t A_t^\top\|_F^2\bigr].
$$

To estimate these expectations efficiently, draw two independent minibatch gradients $g_t^{(1)}$ and $g_t^{(2)}$ at each frozen checkpoint. For each pair, compute

$$
\frac12\|g_t^{(1)}-g_t^{(2)}\|_F^2
$$

and

$$
\frac12\left(
\|B_t^\top(g_t^{(1)}-g_t^{(2)})\|_F^2+
\|(g_t^{(1)}-g_t^{(2)})A_t^\top\|_F^2
\right).
$$

Average each quantity over the independent pairs. These are unbiased estimates of the two conditional variances and use exactly the same diagnostic pairs. Their ratio is not itself an unbiased estimator of the variance ratio. Undefined ratios are recorded as missing.

The corresponding bound, already used in the manuscript, is

$$
\mathbb E\bigl[\|B_t^\top\xi_t\|_F^2+\|\xi_t A_t^\top\|_F^2\bigr]
\le
(\|A_t\|_2^2+\|B_t\|_2^2)\mathbb E\|\xi_t\|_F^2.
$$

Panel (c) divides the estimated factor-gradient variance by the estimated weight-gradient variance and plots $\|A_t\|_2^2+\|B_t\|_2^2$ as its upper bound. Panel (d) divides that ratio by $\|A_t\|_2^2+\|B_t\|_2^2$, giving an upper bound of one. The figures introduce no additional symbols for these quantities. They also do not label measured variance as $\sigma^2$, which is the uniform upper bound in the manuscript.

The code checks this inequality for the actual diagnostic samples. The ambient-weight variance is computed using small minibatch Gram matrices, avoiding a full 768-by-3072 gradient matrix when that is cheaper. Noise-norm matrix products and reductions use double precision; the forward pass and residual derivatives use the selected training precision (float32 by default). Diagnostic randomness is separate from training randomness.

Full training and held-out mean factor gradients are evaluated separately at every checkpoint. Their squared norms and the squared norm of their vector difference are saved. These are instantaneous full-split measurements. They are not minibatch-gradient norms, all-iterate cumulative averages, or exact population gradients. Training and test losses are also full-split means at checkpoints.

The CSVs save Monte Carlo standard errors for the paired variance estimates. Figure bands instead show one sample standard deviation across the reported seeds. For a publication run, compare `--diagnostic-pairs 32` against 128 on the same seeds and check the saved standard errors before settling on a diagnostic budget. Changing diagnostic settings does not change the training draws.

## Running on your AutoDL machine

Extract `ViT_Noise_EXP.zip` into `/root/code/LoRASGD`, creating a sibling of the existing `ViT_EXP` folder. The three Python files are independent of the old plotting schema.

If your existing ViT cache has finished preparing, use it directly:

```bash
cd /root/code/LoRASGD/ViT_Noise_EXP
python3 ViT_Run.py \
  --cache /root/code/LoRASGD/ViT_EXP/data/vit_cifar100.npz \
  --device cuda &&
python3 ViT_Plot.py
```

To use a manually chosen constant learning rate, add `--lr 0.01` to the run command. No recalibration is performed.

If the cache has not been prepared, finish the existing `ViT_Prepare.py` extraction, or use the copy in this folder. With the dataset and model already stored locally, no download is needed:

```bash
python3 ViT_Prepare.py \
  --data-dir /root/autodl-tmp/LoRASGD/datasets \
  --model /root/autodl-tmp/LoRASGD/models/vit-base-patch16-224-in21k \
  --device cuda
```

The model argument above must be the actual local directory containing the checkpoint files. With a working Hugging Face connection, omit `--model` and add `--download` to allow the default checkpoint to be retrieved. This revised experiment still requires the pretrained model once; it does not solve a blocked model-download connection. The preparation script reuses an existing cache and never needs to run again for a new diagnostic budget or learning rate.

A short pipeline check on the real cache is available before the full run:

```bash
python3 ViT_Run.py --device cuda --lr 0.01 --seeds 13 \
  --steps 20 --checkpoints 2 --diagnostic-pairs 4 --diagnostic-batches 16 &&
python3 ViT_Plot.py
```

This short check is not a scientific result. Each run gets a new results directory. To plot a particular completed run later, use `python3 ViT_Plot.py --run-dir /absolute/path/to/run`.

## Outputs

- `experiment.json`: setup, selected rate, pilot scores, sampling and diagnostic metadata, completion status.
- `ViT_trajectory.csv`: full losses, current spectral factor size, instantaneous full training/test squared factor-gradient norms, and their squared vector gap.
- `ViT_noise.csv`: paired variance estimates, standard errors, amplification, and normalized amplification at each checkpoint and diagnostic batch size.
- `figures/ViT_training.png` and `.pdf`: full training loss, spectral factor growth, and instantaneous full training/test stationarity.
- `figures/ViT_noise.png` and `.pdf`: the two estimated noise variances, their ratio with upper bound $\|A_t\|_2^2+\|B_t\|_2^2$, and the ratio divided by this bound.

## Interpretation for the paper

If measured amplification rises as the factors grow, this supplies empirical evidence that factor geometry affects the noise encountered by LoRA-SGD. If the weight noise decreases while factor noise rises or remains substantial, the experiment directly illustrates why a bound on ambient gradient variance alone is not the same as a bound on factor-gradient variance. A different observed trend must be reported as such: the inequality does not assert monotonic amplification.

The ratio in panel (d) being below one is a check of a mathematical inequality and its implementation. The substantive empirical findings are the magnitude of the noise-variance ratio, its evolution along actual training, and how informative or loose the bound involving $\|A_t\|_2^2+\|B_t\|_2^2$ is. Under IID sampling with replacement at a fixed iterate, expected minibatch variances are inversely proportional to the diagnostic batch size, providing an additional consistency check.

These ordinary learning rates are not certified to satisfy the theorem's conservative step-size cap. This experiment illustrates the mechanism motivating factor control. It does not establish the T^(-1/4) convergence rate, validate high-probability coverage, prove a global variance assumption, or establish the population generalization bound. Test-set gradients remain held-out finite-sample measurements.

Suggested prospective methods paragraph:

> To examine the role of factor growth in stochastic-gradient noise, we train a rank-4 adapter in the final MLP projection of a pretrained ViT on CIFAR-100 using a constant learning rate. At fixed checkpoints, we estimate $\mathbb E\|\xi_t\|_F^2$ and $\mathbb E[\|B_t^\top\xi_t\|_F^2+\|\xi_t A_t^\top\|_F^2]$ using independent pairs of minibatch gradients. We compare their ratio with $\|A_t\|_2^2+\|B_t\|_2^2$, varying the diagnostic batch size while keeping the training trajectory fixed.

## Validation of this code package

The training/plotting pipeline and analytic calculations were checked on small synthetic cached inputs. The checks covered factor gradients against finite differences, both noise-norm calculation paths against explicit matrices, paired-estimator unbiasedness by exhaustive finite-sample enumeration, and identical final factors after changing only diagnostic settings. Actual CIFAR-100/ViT GPU training has not been run in the development environment, and this package contains no experimental results. Use the saved pilot scores and diagnostics to assess the real runs before writing a results paragraph.
