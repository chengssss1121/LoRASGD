# Hyperparameters and provenance

This document separates values reported in Mu and Klabjan's paper from choices introduced for the present spectral-growth experiment. The source is Section 4 and Appendix B, Table 1, of *On the Convergence Rate of LoRA Gradient Descent*.

## Settings reported for the image experiments

| Setting | Logistic regression on ResNet-18 embeddings | ResNet-18 on CIFAR-10 images |
| --- | --- | --- |
| Dataset | CIFAR-10 | CIFAR-10 |
| Classes | 10 | 10 |
| Loss | Cross-entropy | Cross-entropy |
| Input dimension | 512 | `32 x 32 x 3` |
| LoRA rank | 4 | 20 |
| Minibatch size | 512 | 512 |
| Epochs | 60 | 250 |
| Constant learning rates | `0.03, 0.05, 0.07` | `0.03, 0.04, 0.045` |
| Original adaptive alpha | 1 | Not reported/tested for this model |
| Loss-based adaptive alpha | 0.8 | 15 |
| Normalized alpha | 0.06 | 0.1 |
| Base weights | Randomly initialized and frozen for the adapted model | Randomly initialized and frozen |
| Adapter initialization | Random A, zero B | Random A, zero B |
| LoRA rank scaling | None | None |
| BatchNorm handling | Frozen extractor in evaluation mode | Running statistics frozen using `model.eval()` |

The logistic inputs are produced by an ImageNet-1k-pretrained ResNet-18. The base classifier that receives those inputs is a separate frozen, randomly initialized matrix. These two weight sources must not be confused.

The paper also reports a longer 1,000-epoch logistic run to investigate continued factor growth. Its TinyLlama/Alpaca experiment has a different architecture and rank/batch-size configuration and is not implemented in this image-framework package.

## Settings selected in this project

| Parameter | Primary choice | Reason/status |
| --- | --- | --- |
| Optimizer | Simultaneous SGD | User's requested update; both gradients from one pre-update state |
| Momentum | 0 | Preserve ordinary SGD |
| Optimizer weight decay | 0 | Avoid adding factor regularization |
| Gradient clipping | Disabled | Avoid changing the update |
| LoRA scaling | 1 | Match the paper's decision not to scale by rank |
| A initialization | Independent zero-mean Gaussian entries | Explicit reconstruction choice consistent with the random-A/zero-B setup |
| A standard deviation | `0.01` in example commands | Explicit project choice; not claimed as a reported image-experiment value |
| B initialization | All zeros | Standard LoRA initialization |
| Main seed | 13 | Explicit project choice |
| Suggested additional seeds | 29, 47 | Replication design choices |
| Sampling | IID minibatches with replacement | Match conditional-unbiasedness convention in the SGD analysis |
| Data augmentation | None in the default fixed-objective experiment | Avoid adding unrecorded sample randomness |
| Factor logging | Every optimizer update, plus initialization | Track the full finite trajectory |
| Norm calculation | Numerical SVD/eigenvalue diagnostics | Distinguish spectral norm from Frobenius norm |
| Product regularization | 0 | Baseline unregularized classification |
| Optional product regularization | `0.001` | Separate coercive-loss probe; not optimizer weight decay |
| Theorem diagnostic delta | `0.05` | Choice for a conservative theory-motivated schedule |

Every actual run's `config.json` is authoritative if a CLI override differs from these examples. Do not substitute the example seed or standard deviation for a value recorded in a real run.

## Learning-rate scope

The primary `constant` schedule uses one scalar learning rate for all LoRA factors. It does not implement the original paper's adaptive or normalized formulas, even though those formulas are listed above to document the reference framework. A number such as `alpha_norm=0.06` is not an interchangeable constant SGD learning rate.

The optional `theory-general` classifier schedule uses the draft's fixed-horizon cap. It depends on the planned number of updates and the recorded loss/gradient constants. A practical rate from the paper's sweep can be much larger. Keep those two experiment types separately labeled.

The horizon is measured in updates. With IID sampling and no `max_steps` truncation,

\[
T=\texttt{epochs}\times\left\lceil N/\texttt{batch\_size}\right\rceil.
\]

At a fixed epoch count, changing batch size changes both stochastic averaging and the number of factor updates.

### Constants in the classifier's theorem-step diagnostic

Let `R_x` be the maximum Euclidean norm of a training feature vector. For the cached-feature softmax loss, the code uses the conservative fixed-training-set constants

\[
L=\max\{1,R_x^2/2+\lambda\},\qquad
b=\sqrt2R_x,\qquad
\sigma^2=2R_x^2.
\]

The softmax gradient has the form `(p-e_y) x^T`, whose Frobenius norm is at most `sqrt(2) R_x`. Its Hessian with respect to the weight matrix has operator norm at most `R_x^2/2`. The gradient of the product regularizer is deterministic at a fixed update matrix and cancels from the stochastic noise. These facts give the displayed conservative bounds; an IID minibatch has no larger variance than the single-sample upper bound used here.

With the initial empirical regularized loss `J_S(B_0,A_0)` and the measured initial spectral sum `S_0`, the code sets

\[
C_0=1+J_S(B_0,A_0)+S_0,
\qquad
\eta=\frac{1}{64\bigl((b+8L\sqrt r\,C_0)^2+2\sigma^2\bigr)
T^{3/4}\log(4T/\delta)}.
\]

The measured maximum feature norm gives a bound for the fixed empirical training problem. It is not, by itself, a uniform bound over an unseen population distribution. The reference line is logged as a theorem-step diagnostic only when the actual rate satisfies the cap and sampling is IID. The saved configuration always reports the reference cap and whether the recorded schedule matches the required step/sampling conditions.

## Data and preprocessing choices

The feature extractor uses Torchvision's documented ResNet-18 ImageNet weights and deterministic preprocessing. Those transforms resize and crop images to the checkpoint's expected input convention before extracting the 512-dimensional pooled features. This is an explicit, reproducible implementation choice; the paper excerpt does not fully specify the extraction pipeline.

The raw-image ResNet experiment uses its documented fixed CIFAR-10 preprocessing. Normalization, input resizing, pretrained versus random base weights, and BatchNorm behavior must be recorded because they can materially change the loss and the norm trajectory.

The implementation injects adapters into all convolutional and linear layers, whereas the paper specifically describes adapters on the convolutional layers. Adapting the final linear classifier is an explicit reconstruction choice. Its configured rank can exceed the smaller dimension of that classifier; the factor width is still well defined, but it is then overparameterized rather than a strict rank reduction. The default uses Torchvision's standard ResNet stem; `--cifar-stem` selects a different architecture and must be reported.

For a direct comparison of learning rates or ranks, reuse the same feature file, seed convention, frozen base initialization convention, and sampling scheme. For independent repetitions, vary the seed and report how it controls both initialization and minibatch draws.

## Suggested experiment matrix

| Purpose | Rank(s) | Learning rate(s) | Epochs | Seeds |
| --- | --- | --- | --- | --- |
| Paper-inspired logistic baseline | 4 | `0.03, 0.05, 0.07` | 60 | `13,29,47` |
| Rank dependence | `2,4,8` | `0.05` initially | 60 | `13,29,47` |
| Continued growth after apparent loss stabilization | 4 | Selected stable baseline rate | Up to 1,000 | At least the baseline seeds |
| Product-regularized comparison | 4 | Same as matched unregularized run | 60 | Matched seeds |
| Conservative theorem-step diagnostic | 4 | Computed `theory-general` rate | Specified fixed horizon | Matched seeds |
| Optional ResNet baseline | 20 | `0.03,0.04,0.045` | 250 | `13,29,47` |

Inspect stability before launching long sweeps. Report unstable settings as outcomes rather than silently replacing them with easier configurations.

## Required reporting fields

Record the dataset/cache identity, model/checkpoint, code and package versions, seed, rank, A standard deviation, B initialization, scaling, batch size, sampling scheme, number of updates, learning-rate rule, momentum, clipping, weight decay, product regularization, dtype, device, BatchNorm mode, preprocessing, and logging interval. For multi-adapter models, record which layers are adapted and the exact aggregation used in plots.

Official API references: [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html), [singular values](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svdvals.html), and [ResNet-18 weights/transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
