# Precision

`btr.IMPALA_DTYPE` runs the trunk's convolutions in bfloat16. Everything else, including the
parameters and Adam's moments, stays float32. `ImpalaCNNLarge`'s docstring says which parts of
the computation the dtype reaches.

## Throughput

1237 -> 2152 env steps/s, the largest single win the profiling found (see `performance.md`).
float16 measures the same and is not used, since the residual stream has no loss scaling behind
it and bfloat16 is the one that keeps float32's exponent range.

## Over a full budget

Settled over a full budget. Two runs matched step for step:

- `logs/breakout_2026-09-12_02-33-13`, float32, to 951,987 gradient steps
- `logs/breakout_2026-09-13_04-18-22`, bfloat16, to 857,335 gradient steps

Over the 10M-55M env-step overlap, bfloat16 scores 513 +/- 74 mean evaluation score against
float32's 480 +/- 68 (Welch p = 0.002; one seed each, so read this as no worse rather than
better), at 2020 against 1063 env steps/s. `spectral_sigma` and `noisy_sigma` agree between the
two to better than 3% at 52M env steps, and `dormant_frac` and `dead_frac` stay at zero in both.

## Per batch

Against an identical float32 net, the trunk's features carry 0.7% relative L2 error and its
weight gradients 0.9% median (1.5% worst). Both are a small multiple of bfloat16's own 0.39%
rounding unit, so the error accumulates over layers but is not amplified.

Nothing compounds by construction either: the parameters and Adam's moments are float32, so no
rounding accumulates in the weights, and bfloat16 keeps float32's exponent range, so nothing
underflows.

## Tensor cores

A 16-bit dtype is what cuDNN's tensor cores read. Given float32, cuDNN runs them anyway, on a TF32
copy it stages with a conversion pass per convolution. That costs a sixth of device time and more
than doubles what the convolutions themselves take.
