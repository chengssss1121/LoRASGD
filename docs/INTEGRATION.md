# Add exact factor logging to an existing PyTorch or PEFT run

The reusable logger measures raw stored matrices. For every pair it records

\[
\|A_t\|_2^2,\qquad \|B_t\|_2^2,\qquad
\|A_t\|_2^2+\|B_t\|_2^2.
\]

All norms are calculated in CPU float64 with singular value decompositions.
These are numerical spectral norms, not power-iteration estimates or Frobenius
norms. Logging does not run a forward pass, call backward, change train/eval
mode, alter gradients or parameters, or consume random numbers. Device-to-CPU
copies synchronize with accelerator work, so logging every step can be costly.

## Simultaneous SGD: one backward pass and one optimizer step

Run from the project root so `lora_growth` is importable. In a standard LoRA
model freeze the pretrained weights, biases, normalization layers and any
classification head before creating the optimizer. Train only A and B when
matching the factor-only mathematical setup.

```python
import torch
from lora_growth.torch_logger import FactorGrowthLogger

# `model` and `loader` are created by your existing training code.
# Custom layers in this package use `.lora_A` and `.lora_B` Parameters.
# Standard PEFT models use `...lora_A.<adapter>.weight` and B analogously.
for name, parameter in model.named_parameters():
    parameter.requires_grad_("lora_A" in name or "lora_B" in name)

parameters = [p for p in model.parameters() if p.requires_grad]
if not parameters:
    raise RuntimeError("No trainable LoRA A/B factors were found")
optimizer = torch.optim.SGD(
    parameters, lr=0.03, momentum=0.0, weight_decay=0.0
)

# Training mode is an experimental choice. Keep BatchNorm in eval mode for
# the paper's ResNet experiment, as train_resnet.py does.
completed_updates = 0
with FactorGrowthLogger("runs/my_existing_run/factor_logs", log_every=1) as logger:
    logger.log(model, step=0, epoch=0, lr=optimizer.param_groups[0]["lr"])
    for epoch in range(number_of_epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model, batch)  # Your existing scalar loss.
            loss.backward()                  # Gradients of BOTH factors at old state.
            optimizer.step()                 # Update BOTH factors together.
            completed_updates += 1
            logger.log(
                model,
                step=completed_updates,
                epoch=epoch + 1,
                lr=optimizer.param_groups[0]["lr"],
                batch_loss=loss.detach().item(),
            )
    # With log_every=1 the final step was already logged. With a larger cadence,
    # force the final state only if it was not on the logging cadence.
    if completed_updates % logger.log_every != 0:
        logger.log(model, step=completed_updates, epoch=number_of_epochs, force=True)
```

Do not call a second backward pass or recompute B's gradient after updating A.
That would create an alternating update rather than simultaneous SGD. The logged
`step` is the number of completed optimizer steps. Step 0 is the initialization;
after k updates the record is step k (the paper's factors indexed from 1 would
be A_(k+1), B_(k+1)). `batch_loss` is the loss used for the just-completed update,
not an extra forward evaluation of the post-update state.

## Hugging Face PEFT

The logger has no PEFT dependency. It discovers standard LoRA linear and
convolutional factor dictionaries by their `lora_A` and `lora_B` interfaces and
logs every configured adapter, including inactive adapters. The `adapter_active`
column identifies which adapters were active when logging. Aggregates include
all configured pairs; filter the pair CSV if you want only active adapters.

For a standard LoRA configuration with `use_rslora=False`, scaling is
`s=lora_alpha/r`. To match the unscaled update W0+BA use `lora_alpha=r`,
`lora_dropout=0.0`, `bias="none"`, and disable DoRA and other LoRA variants.
PEFT may make task-specific heads trainable; the filtering shown above freezes
them. Initialize A explicitly with the chosen Gaussian standard deviation and
B to zero if that is your protocol; PEFT's default initializer is not the same
as an arbitrary Gaussian initialization.

The logger records:

| Columns | Meaning |
| --- | --- |
| `a_spectral_sq`, `b_spectral_sq`, `spectral_sum` | Raw A/B quantities, independent of the LoRA scaling parameter |
| `product_spectral`, `product_frobenius` | Norms of the raw product BA |
| `adapter_scaling` | Stored scalar s for the adapter |
| `scaled_product_spectral`, `scaled_product_frobenius` | Norms of the individual standard-LoRA contribution sBA |
| `adapter_active` | Whether this configured adapter is active |
| `sampled_running_max_spectral_sum` | Largest recorded raw spectral sum for this pair so far |

The scaled-product columns do not describe an entire multi-adapter sum or a
merged pretrained matrix. Embedding LoRA and DoRA are rejected explicitly.
Flattened convolutional factors are accepted when their matrix dimensions are
compatible; grouped or unconventional factorizations require their own metric
definition. Standard custom LoRAConv2d in this package supports groups=1.

## Output and aggregation

`pair_metrics.csv` has one row per pair and logged step.
`aggregate_metrics.csv` has one row per logged step and keeps two distinct
statistics: `sum_spectral_sum` sums over all pairs, while
`max_pair_spectral_sum` takes the maximum across pairs. They are not the same
as the single-pair quantity in the theorem. The aggregate running-max fields
are named `sampled_running_max_sum_spectral_sum` and
`sampled_running_max_pair_spectral_sum`.

With `log_every=1`, running maxima cover every optimizer-step state. With a
larger interval they cover only sampled states and can miss an intermediate
spike. The logger refuses to overwrite existing CSVs, requires initialization
at step 0, and requires strictly increasing logged steps and fixed factor shapes.
Use a new output directory for each independent run.

## Interpretation

A flat curve is empirical evidence about the measured run and finite horizon;
it does not prove a uniform bound for all horizons, all samples, or all seeds.
A generic Transformer need not satisfy global samplewise L-smoothness, so the
logger is an empirical diagnostic rather than a certification of the theorem's
assumptions or constants. Comparing seeds and horizons should keep the frozen
base model, data split, factor initialization convention, sampling protocol and
learning-rate schedule fixed and explicitly recorded.

Official API references:

- [PyTorch SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)
- [PyTorch singular values](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svdvals.html)
- [PEFT LoRA configuration](https://huggingface.co/docs/peft/en/package_reference/lora)
