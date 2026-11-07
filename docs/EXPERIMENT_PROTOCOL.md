# Experiment protocol

## Scientific question

Measure how the two low-rank factors evolve during simultaneous LoRA-SGD. The primary quantity for one matrix pair is

\[
S_t=\|A_t\|_2^2+\|B_t\|_2^2.
\]

The study separates three observations: factor growth, product growth, and optimization progress. A small product norm alone does not imply small factor norms because `(B,A)` and `(cB,A/c)` represent the same product. A decreasing loss alone does not imply bounded factors.

The experiment adapts Mu and Klabjan's CIFAR-10 classification framework to this question. The original paper plots stacked-factor/Frobenius quantities and examines several learning-rate schedules. This package explicitly logs spectral quantities and uses ordinary simultaneous SGD as its primary update.

The classifier's `--synthetic` option and the vision scripts' `--fake-data` options are execution checks using generated data. Their outputs must remain labeled as smoke data and must not be included in a CIFAR-10 empirical result table.

## Model and objective

For cached features `x_i` and class labels `y_i`, the trainable classifier has a frozen base matrix `W0` and update `BA`, with `A` of shape `(r,d)` and `B` of shape `(K,r)`. The logits use the selected implementation's frozen bias and the effective weight `W0+BA`. The empirical loss is mean multiclass cross-entropy, optionally augmented by

\[
\frac{\lambda}{2}\|BA\|_F^2.
\]

The regularizer acts on the weight update, not separately on the two factors. Set `lambda_product=0` for the unregularized experiment closest to the paper's classification setup. A positive value defines a separate regularized experiment and helps enforce bounded sample-loss sublevel sets in the update matrix.

The optional ResNet experiment injects low-rank updates into model layers. Frozen base weights and frozen BatchNorm running statistics are recorded implementation choices. Multi-layer nonlinear network training should be presented as an empirical extension, not as a direct verification of a global smoothness assumption.

## Simultaneous update semantics

For a minibatch weight-gradient `g_t`, both factor gradients are evaluated at the same current pair:

\[
A_{t+1}=A_t-\eta_t B_t^\top g_t,
\qquad
B_{t+1}=B_t-\eta_t g_t A_t^\top.
\]

The NumPy trainer calculates both gradients, creates both new arrays, and then replaces the current pair. The PyTorch trainer performs one backward pass followed by one optimizer step over all factor parameters. Gradients are not recomputed between the updates of `A` and `B`.

Primary settings: zero momentum, zero optimizer weight decay, no gradient clipping, no LoRA dropout, no factor normalization, and LoRA scaling equal to one. A change to any of these settings must appear in the experiment label and saved configuration.

## Sampling and time

- `iid`: each update draws an independent minibatch with replacement from the fixed training set. This directly gives conditional unbiasedness of the batch gradient. One nominal epoch consists of `ceil(N/batch_size)` updates.
- `shuffle` in `train_logistic.py`, or `epoch` in `train_resnet.py`: each epoch shuffles the data and visits it without replacement. This is common in practice but differs from the independent-index model used in the draft.
- `step=0` is the initialized state. It corresponds to paper index `t=1` in the draft.
- A record at `step=k` contains the factor state after exactly `k` updates.
- The planned horizon is the number of optimizer updates, not the number of examples and not the number of epochs.

Changing batch size changes the number of updates at a fixed epoch count. Compare both update-matched and example-matched views when discussing batch-size effects. Record the actual horizon if a `max_steps` limit truncates training.

## Diagnostics

| Column | Definition |
| --- | --- |
| `a_spectral_sq` | Largest singular value of `A`, squared |
| `b_spectral_sq` | Largest singular value of `B`, squared |
| `spectral_sum` | `a_spectral_sq + b_spectral_sq` |
| `a_frobenius_sq` | Sum of squared entries of `A` |
| `b_frobenius_sq` | Sum of squared entries of `B` |
| `frobenius_sum` | `a_frobenius_sq + b_frobenius_sq` |
| `product_spectral` | Largest singular value of `BA` |
| `product_frobenius` | Frobenius norm of `BA` |
| `imbalance_spectral` | Spectral norm of `AA^T-B^TB` |
| `rank` | Configured factor width `r`, not a numerical rank estimate |

Diagnostics use numerical linear algebra rather than finite-step power iteration. “Exact” means no power-iteration approximation; ordinary floating-point rounding remains. Product diagnostics use the reduced factorization of the two factors to avoid allocating the full product when possible. The `rank` column should not be interpreted as the number of nonzero singular values.

Log initialization, every optimizer update for the primary runs, and the final state. If diagnostic cost requires a larger logging interval, distinguish sampled maxima from maxima over all iterates. Do not silently interpolate a sampled curve into a claim about unobserved iterates.

For multiple adapters indexed by `ell`, report at least selected individual curves and define any aggregate explicitly:

\[
S_t^{\rm sum}=\sum_\ell S_t^{(\ell)},
\qquad
S_t^{\rm max}=\max_\ell S_t^{(\ell)}.
\]

Neither aggregate is automatically the single-pair theorem quantity. Summing pairwise spectral norms also differs from the spectral norm of a hypothetical concatenated matrix.

## Factor-growth theorem diagnostic

The general factor-growth bound in the draft uses a fixed terminal horizon:

\[
C_T=2\bigl(\|A_0\|_2^2+\|B_0\|_2^2\bigr)+T^{1/4}.
\]

The shift from `A_0,B_0` here to `A_1,B_1` in the paper is indexing only. Compare the entire measured trajectory through the terminal update with this one horizontal threshold. A pointwise curve using `t^(1/4)` is a different claim.

A bound comparison is meaningful only when the experiment satisfies the accompanying step-size and loss assumptions. The theorem-motivated schedule records its chosen constants and rate. Practical constant learning rates are independent empirical probes and should be labeled accordingly. Failure to violate a conservative bound is empirical consistency, not a proof of the bound or its probability.

The synthetic and cached-feature classification experiments have explicit, fixed sample losses. A finite feature cache gives finite bounds on the input norms. Multiclass cross-entropy with a linear weight matrix has a globally Lipschitz gradient; adding the quadratic product regularizer preserves smoothness as a function of the update matrix. The corresponding factorized loss is not globally smooth in both factors merely because the matrix loss is smooth.

## Recommended comparisons

1. Run the paper-inspired logistic baseline with rank four and the three constant learning rates, for at least three seeds.
2. Plot the two individual squared spectral norms and their sum. Include initial values and use the same axis convention across runs.
3. Repeat at ranks `2,4,8` without changing the initialization convention silently. Record whether the standard deviation or a norm-normalized initialization is being held fixed.
4. Compare an unregularized run with a product-regularized run. Treat this as an objective change and report its value of `lambda_product`.
5. Extend a selected run beyond sixty epochs if the loss appears stable while the factors still grow. A bounded finite trace cannot determine the infinite-time behavior.
6. Use the optional ResNet experiment as a practical multi-layer check, with layerwise plots and frozen BatchNorm statistics.

The listed seeds or ranks are a suggested experimental design, not fabricated observations. Inspect actual CSV files before writing claims about monotonicity, stability, or convergence.

## Reporting template

> We trained the two LoRA factors simultaneously using minibatch SGD with [sampling scheme], batch size [value], learning rate [value/schedule], zero momentum, and no factor weight decay or clipping. The frozen base [model/checkpoint] was adapted with rank [value] and scaling one. We initialized the entries of A from a zero-mean Gaussian with standard deviation [value] and initialized B to zero. We recorded the largest singular value squared of each factor and their sum [logging cadence], starting before the first update. Curves show [individual seeds/mean with stated interval] over [number] seeds. Norms are [per adapter/explicit aggregation].

If a plot uses smoothing, report its window and retain the raw trajectory. When averaging seeds, state whether bands show standard deviation, quantiles, or confidence intervals; they are different quantities. Report failed runs and their configurations rather than discarding them from the average.

## Scope of the evidence

Finite traces do not prove an asymptotic power law, global boundedness, a stationarity theorem, or a generalization theorem. The exponent in a worst-case upper bound need not appear in the observed norm trajectory. The theorem's spectral quantities and the reference paper's Frobenius quantities can be compared descriptively, but they are not interchangeable.

