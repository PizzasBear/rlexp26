# Plasticity, scale and the loss

Measurements behind the plasticity, weight-decay, Huber-knee, gradient-clip and XQC entries in
`btr.py`'s backlog, and behind the diagnostics in `utils.py`. Moved out of the code's comments on
2026-09-24.

## NoisyNets' sigma drifts

Over two full runs, and unaffected by PER_BETA:

- every NoisyLinear's `noisy_sigma_neg_frac` moves from 0 to 0.37-0.48, and its
  `noisy_sigma_signed` collapses to zero;
- every sigma tensor's `adam_snr` sits below the 0.18 iid-gradient floor, and its `adam_step_rel`
  rises (x3 on `value_linear1/kernel_sigma`, since sigma's RMS shrinks underneath it);
- `noisy_sigma_ratio` ends at 0.024 and 0.071 on the two output layers, so by then the head is
  not exploring with this noise.

Both candidate fixes are on the optimiser side: weight decay on the sigma parameters, or BTR's one
draw per gradient step in place of NoisyLinear's per-`__call__` redraw. The per-call redraw is what
makes the forward passes in a train_step disagree about eps, and it is why this run is more exposed
than BTR is. The schedule is not the problem.

## The pathology is scale, not dormancy or rank

Two full runs read:

- `dead_frac` is zero throughout.
- `dormant_frac` is non-zero only before 0.31M env steps.
- `effective_rank` is bounded by the batch rather than the trunk (see below); a trained trunk
  reads above a fresh one on the same observations.

What is unbounded is scale, and it eats into the denominator of the effective learning rate:

- `weights/global_norm` goes 61 -> 162 and is still rising linearly at the end.
- `value_linear1/kernel_mean`'s RMS goes x11, and `quantile_embedding/linear/bias` x89.
- `adam_snr` stays flat, so the output layers' `adam_step_rel` falls x6 (3.1e-4 -> 4.6e-5) with
  no change in the gradient.

That is the ELR collapse Lyle et al. and XQC (ICLR 2026 submission) describe. For scale, a
50M-step run is ~824k updates, which is ~21 resets at BBF's 40k cadence.

## Which layers are scale-invariant

Measured by scaling each kernel and reading the output:

- SpectralNorm already makes all 12 residual convolutions exactly scale-free.
- With `LAYER_NORM` on, so are the three stage conv0s.
- `quantile_embedding`, `value_linear0` and `advantage_linear0` are scale-free to within 1%.

Only `value_linear1` and `advantage_linear1` are left. They carry the scale of a return, which is
also where the x11 growth is, and a categorical head would make them projectable. So most of
`weights/global_norm`'s rise is gauge, and XQC's per-step projection to the unit sphere removes it
where decay cannot. Decaying `quantile_embedding` is what annealed IQN's tau modulation in the
LN + WD run.

## Biases and normalisation gains

Biases grow by one to two orders of magnitude over a full run. With the trunk normalised per
position, the Impala gains form a near-constant pedestal ten times the L2 of every other parameter
put together. That is why `unnormalised_param_mask` keeps them out of the global norms.

## The Huber knee

`IQN_HUBER_LOSS_K` = 1.0 against a raw |td| that spans 0.02-0.45 over a run puts the loss in the
quadratic branch nearly always, so the head fits expectiles. QR-DQN's footnote gives the origin of
the 1.0: DQN clipped the squared error to [-1, 1], which is exactly a Huber with kappa = 1, and the
value has been inherited ever since. It is tied to reward clipping's [-1, 1], not to anything about
the scale of Q.

Two things follow.

- **The estimand is wrong.** IQN reads Q as the mean over uniform tau of the learned statistics.
  For quantiles that is E[Z] by construction; for expectiles it is not. On a return-shaped, skewed
  distribution the kappa = 1 fit comes out biased upward, the direction a max-over-actions bootstrap
  compounds, where kappa = 0.1 does not.
- **The estimand drifts.** kappa is an absolute threshold against a raw |td| that moves 18.5x over
  a run and a Q that moves 200x, so the statistic being fitted at 1M steps is not the one fitted
  at 800k.

kappa = 0 is the pinball loss proper, and it is a real configuration, not a failure mode. QR-DQN
reports it as QR-DQN-0: 199% median / 881% mean human-normalised, against QR-DQN-1's 211% / 915%
over 57 games. It costs ~6%, and it is what the unbiased estimand actually requires. Expect two
things from it:

- Mean |dl/dprediction| at the optimum rises x1.89 against kappa = 1, which at a median grad_norm
  of 4.4 puts most steps into `GRADIENT_CLIPPING_MAX_NORM`.
- The gradient no longer shrinks as the prediction approaches the target, which is one of the two
  candidate mechanisms for the sub-floor `adam_snr`.

## Gradient clipping

Engaged on 8.7% of steps over the last full run, so it is part of the update rule rather than a
rare guard. Adam already bounds each coordinate's step to ~lr regardless of gradient size, so the
clip looks redundant. It is not, because it sits before Adam and protects nu: a spike admitted
into nu is held for 1 / (1 - ADAM_B2) = 1000 steps while mu forgets it in ten, which suppresses
every legitimate update in between. The other reading of 8.7% engagement is that the clip is what
holds `adam_snr` where it is, which is why removing it has to be measured.

## adam_snr below its floor

BTR's `ADAM_EPS` of 0.005 / batch is ~2e-5, four thousand times optax's default, and large enough
to dominate the denominator on a wide layer whose gradient is spread thin. It does: at the end of
a full run, 49% of `advantage_linear0`'s elements have sqrt(v) below eps and 95% below 10 eps.
That layer reads 0.067, where dropping eps reads 0.109. The trunk's convolutions have no element
anywhere near eps and read the same either way.

## effective_rank reference points

On 256 Breakout observations, the batch is what binds:

| states | fresh trunk | trained trunk |
|---|---|---|
| from a random policy | ~5.5 | ~5.5 |
| the trained policy visits | 11.7 | 23.4 |
| synthetic uniform noise | 191 | 140 |

The ceiling is 255. So a low absolute number reflects the input manifold, a fall against a fixed
batch is capacity loss, and training raising the number on its own state distribution is the
healthy case. The uncentred version read ~1 on a freshly initialised network and saturated around
9 whatever the true rank was.

Counting dormancy per (position, channel) pair instead of per feature map reads ~2.4x high at
initialisation on Breakout, and it is comparable to no published figure.

## BatchNorm without CrossQ's joined pass

XQC's Hessian eigenspectra put BN an order of magnitude below LN on condition number. XQC has a
target network too (EMA momentum 0.005-0.01, every step), so CrossQ's joined forward pass does not
follow from having no target network. What it repairs is a mismatch between the replay action
distribution and pi_phi(s'), and the action is not a network input here. On a Zaxxon checkpoint,
obs and next_obs agree to 0.0006-0.0020 sd per channel across all 36 trunk layers. So the joined
pass buys nothing, and BN costs one normalisation layer rather than a 2x trunk pass. BatchRenorm
is CrossQ's answer to minibatch-noise collapses, and it matters more at the smaller batches a
higher replay ratio implies.
