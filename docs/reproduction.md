# Reproduction against BTR

Where this codebase and BTR (arXiv:2411.03820 and github.com/VIPTankz/BTR) differ, and what each
difference was measured to be worth. Moved out of the code's comments on 2026-09-24. See also
`behaviour-policy.md` for the acting and evaluation-policy differences.

## Breakout scores

Checked against BTR's own `results.csv` (200 Breakout evaluations, 100 episodes each), from the
212M-frame run in `logs/breakout_2026-09-13_14-12-41`: 557 against their 549 past 40M frames, and
584 against their 601 over the closing evaluations. The whole remaining shortfall is before 40M
frames, 253 against 383. This run takes off about 5M frames later than BTR, then converges onto
its curve by 15M. PER_BETA is not the cause.

## PER_BETA

BTR's `PER.py` computes its IS weights as `(capacity * prob) ** -self.alpha`, with alpha standing
where beta belongs, so its effective beta is 0.2. The comment beside that line calls it an
accident kept because it performed better. `self.beta` itself is vestigial: initialised to 0.4,
reset to 0 by every insert, and read by no live code.

Sampling frequency times IS weight goes as p ** (PER_ALPHA * (1 - PER_BETA)). At 1.0 that
exponent is zero, so prioritisation reaches only which transitions arrive together. At 0.2 it is
0.16, and prioritisation reaches their expected contribution too. That was worth +50 mean
evaluation score per matched 2M-env-step window (17 of 21 windows) over a pair of full runs. It is
also why `train/grad_norm`'s median is 4.4 rather than 2.7, and why the batch exceeds
`GRADIENT_CLIPPING_MAX_NORM` on 8.7% of steps rather than 0.9%. BTR clips at the same 10 with the
same effective beta.

## Epsilon decay shape

BTR's `EpsilonGreedy.update_eps` is `eps -= (eps - eps_final) / eps_steps`, applied once per
environment step. That takes a constant fraction off the gap above the floor each step, not a
constant amount. Table D6's "epsilon-greedy Decay: 8M Frames" reads like an anneal length, and the
paper never says otherwise, so a linear reading is the natural one. It is wrong by 37x at 8M
frames, where the geometric schedule is at 0.374 against a linear 0.010. The geometric schedule
does not reach 0.011 until 60M frames, and it takes 1.8x as many random actions over the 100M
frames epsilon is live.

`epsilon_at` writes `(1 - 1/steps) ** n` as its continuous limit. That needs no frameskip (8M
frames and BTR's 2M environment steps are the same duration), and it is exact to 5e-7 relative
error anywhere in a budget.

## Adaptive max pooling

`layers.adaptive_max_pool` uses one (kernel, stride, padding) triple for the whole axis. That is
what BTR's paper describes ("identical to a standard 2D maxpooling layer, but ... automatically
adjusts the stride and kernel size"), but not what `torch.nn.AdaptiveMaxPool2d`, and so BTR's
code, runs: torch varies the window per output cell and overlaps the windows wherever the output
size does not divide the input.

They are different functions, not two spellings of one. On the trunk's 11 -> 6, 11.4% of output
cells differ, the pooled vector moves by 0.40 relative L2, and the mean activation rises 31.5%,
since a max over a superset is never smaller. This one is kept because BTR's Table 2 has pooling
costing score and buying robustness. Dropping pooling entirely scores 406k against BTR's 330k at
evaluation epsilon 0, and 171k against 194k at 0.01. This project scores at epsilon 0, so pooling
less, as torch's version does, is the wrong direction here. Speed doesn't decide it either way:
torch's costs 62.1 us against 55.4 at the training shape [256, 11, 11, 64], some 0.06% of a run's
wall.

## LayerNorm against torch

`ImpalaBlock`'s LayerNorm uses Flax's default epsilon of 1e-6, not `torch.nn.LayerNorm`'s 1e-5,
which is the one place the two libraries disagree to any visible degree. Against a PyTorch
reference, the forward pass and both gradients agree to 1e-5 as it stands, and to 2e-7 with
torch's value passed. That is below what the trunk's own bfloat16 rounds away.

Flax's fused `E[x^2] - E[x]^2` variance is 3% faster per gradient step. It is indistinguishable
from the two-pass estimator until a feature map's mean reaches a few hundred times its spread;
here the ratio is 0.2.
