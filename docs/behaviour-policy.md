# Behaviour policy

Measurements behind `ACT_USE_SOFT_POLICY`, `ACT_TEMPERATURE`, the epsilon constants and the
`train/action_gap`, `train/regret`, `train/gap_bar` and `train/sigma_rank` diagnostics. Moved
out of `btr.py`'s comments on 2026-09-24. The runs themselves are in `runs.md`.

## ACT_TEMPERATURE on Phoenix

What the temperature buys on Phoenix is a delay, not a different endpoint. A frame shift was
fitted to each arm's eval curve over 40-160M frames. Every soft run is the argmax-only run's curve
moved right, and overlays it to within 1-6%:

| ACT_TEMPERATURE | shift | late score (150-200M) vs BTR |
|---|---|---|
| 0.005 | 22.5M frames | 0.968 |
| 0.01 | 42.5M frames | 0.935 |
| 0.03 | 40M frames | 0.910 |
| argmax-only | — | 1.128 |

Sharpening from 0.03 to 0.005 is worth ~18M frames of the delay. The 0.01 run gave the same 20M
straight back to one early value collapse: q fell 3.24 -> 1.97 between 5M and 11M env steps, the
policy went near-uniform, and eval stayed flat to 33M frames. That is the risk a sharper draw
carries. The softmax's self-correction only reopens once the gap falls to O(tau), so the lower
the temperature, the deeper the collapse has to go first. Tuning the temperature is worth about
as much as its own run-to-run variance. If soft acting is used, 0.005 is the value to use.

0.02 is the one arm the shift does not describe, and it is also the one arm that moved
`MUNCHAUSEN_TEMPERATURE` along with it instead of decoupling. No shift overlays it: its residual
is 0.193 against 0.236 unshifted, where shifting cuts the other three by 2.4-3.5x. It is a flat
0.70-0.88 of argmax-only in every band, with no early collapse and the weakest late result of any
soft arm: 0.878 [0.861, 0.948] of the 0.03 run over 100-200M. Moving the loss's temperature is a
different experiment from sharpening the draw, and it did not pay.

## Epsilon-greedy on Phoenix

BTR's own schedule, run once on Phoenix, crosses over:

- Against argmax-only it scores 1.450 [0.915, 1.658] over 1-50M frames and 0.660 [0.497, 0.789]
  over 100-200M.
- Against BTR's own curve it is the only arm here ever to beat it early, at 1.155 over 1-50M,
  where every other arm is 0.21-0.69.

The advantage is gone by the 25-75M band, while epsilon is still live. So it is bought in the
frames where epsilon is large (0.053 at 25M, 0.012 at 50M), and it does not compound. The early
shortfall against BTR is the behaviour policy; the late one is not, and what is left there is the
loss and the architecture.

Epsilon is per-env and independent of the weights, which makes it the one escape a confidently
wrong Q cannot fool; soft sampling is only self-correcting. It is also a different mechanism
from the softmax rather than a blunter version of it, and only exploration regret shows the
difference. At the floor, epsilon takes a non-greedy action on 0.875% of draws, about as often
as NoisyNets' own draw does. But it pays 0.008-0.012 of return per decision against that draw's
0.0007-0.0010, because the action it substitutes is uniform rather than value-weighted
(`results/rho_phoenix_*.json`). That is ten times the price for the same rate. `train/regret`
makes the same comparison in-run against a softmax rather than NoisyNets: 0.0053 at the floor
against a 0.02 draw's 0.0018.

## Scoring with NoisyNet noise live

`Evaluator` scores with NoisyLinear drawing live, where BTR's `prep_evaluation` zeroes
`weight_epsilon` / `bias_epsilon`. That was long the leading candidate for the shortfall against
BTR, and it is not one. On the matched 200M Zaxxon checkpoints, 64 episodes a cell, zeroing the
noise is neutral for an argmax-trained net (1.016, p = 0.34) and costs a soft-trained one 10%
(0.904, p = 0.004). A net scores best under the policy class it was trained under, so scoring
the mean weights is not the free correction it looks like.

## The action gap

`train/action_gap` logs the mean top-two gap every run. On Phoenix, the game BTR's Table 5
reports, this codebase lands inside BTR's own band: 0.179 for the soft-trained net and 0.231 for
the argmax-trained one, against BTR's 0.282 with IQN and 0.180 without. There is no gap deficit.
The old "26x below BTR" was inferred from `policy_entropy` rather than measured, and it compared a
soft Zaxxon run against an argmax Phoenix number.

The gap depends on the game and its action set, by 5.4x: 0.033 on Zaxxon against 0.179 on Phoenix
for two soft runs at the same temperature. Zaxxon's 18 actions are five directions x fire and
alias heavily, so a top-two gap there is largely the margin between two near-duplicates. Phoenix's
8 actions are distinct, so part of that 5.4x is the action set rather than the task. A temperature
calibrated on one game does not carry to another.

The behaviour policy moves the gap 1.24-1.29x, soft below argmax on both games. Epsilon moves it
much further in the other direction: `train/action_gap` converges to 0.146 on the epsilon arm
against 0.256 and 0.286 on two soft ones. `policy_entropy` and `noisy_sigma_ratio` read the same
way, so uniform exploration leaves a network that separates actions about half as well. Every one
of these is measured on the states that run itself visits. How much is the Q function, and how
much is which states epsilon puts in the buffer, is open.

Zeroing the NoisyNet draw moves the gap by 1-6% everywhere, so BTR's `prep_evaluation` is not what
separates the two implementations either.

The gap does not track q. Over a run `train/q` moves ~195x while `train/action_gap` moves ~13x and
is within 2x of its final value from 8M frames on. So gap/q falls ~7x, and a fixed temperature
does not sharpen itself as training proceeds: from 8M frames to the end it gives very nearly a
fixed policy sharpness.

The gap is a small residual of a large common value: it is the difference of two discounted sums
over a 1 / (1 - DISCOUNT) = 333-step horizon, and the dueling head splits the two by construction.
So read it off `train/action_gap`, never infer it from the scale of Q or from an entropy.

## Why regret, not the gap or the entropy

Neither the gap nor the entropy is what a temperature should be calibrated against: both depend
on the action set rather than on behaviour. A duplicated action drives the gap to zero with no
change in behaviour, and actions that are never taken raise log |A| while the entropy stays put.
Regret prices each action by what it is worth. It is also the measure that separates the two
exploration mechanisms, which agree on their off-greedy rate to within a tenth and disagree
tenfold on price.

`train/regret` is blind to one term. The argmax of a NoisyNet draw is the argmax of the q it is
scored against, so a run acting greedily reads 0 while its weight draw still costs 1.0-1.5% of q
at the final checkpoints. `scripts/probe_checkpoint.py --sections regret` measures that offline
with a zeroed reference pass.

The p5/p50/p95 series exist because entropy is an exponential readout of gap over temperature, so
its mean is set by whichever tail is fattest. A temperature calibrated against a mean gap leaves
every state below p5 of the gap drawing near-uniformly.

## gap_bar and sigma_rank

These are the two per-state scales `temperature.tex` builds its rule from. `gap_bar` is the mean
gap, which is the regret of acting uniformly. `sigma_rank` is the spread of the action ordering
across the quantile axis, which is large where the optimistic and pessimistic orderings disagree.
Their ratio is the sharpness that document claims transfers between games: 11.2-11.8 on three
soft-trained checkpoints.

In-run they are estimated over `IQN_TRAIN_SAMPLES` quantiles, against the offline probe's
`IQN_ACT_SAMPLES`. So `sigma_rank` reads ~3.5% low in-run and the ratio ~3.6% high. Correct a level
by that before comparing it to `results/spread_*.json`; a trajectory needs no correction.

## Munchausen Theorem 2

Theorem 2 of Munchausen (arXiv:2007.14430) puts M-VI's converged gap at 1 / (1 - alpha) = 10x that
of the MDP regularised at (1 - alpha) * tau. The paper states (1 + alpha) / (1 - alpha), which is a
slip in the last line of its Appx. A.4: it collects 1 + alpha / (1 - alpha) as
(1 + alpha) / (1 - alpha) rather than as 1 / (1 - alpha). This was verified by iterating M-VI and
entropy-regularised VI to convergence on random finite MDPs, where the ratio reads 10.000000 at
every state-action pair at alpha = 0.9. It was cross-checked against rho_env = (1 - alpha) *
rho_tot, which `temperature.tex` derives from the bonus decomposition instead.

The correction settles which temperature acts. The amplification cancels the (1 - alpha) in the
regularisation coefficient exactly, so softmax(q / MUNCHAUSEN_TEMPERATURE) is the soft-optimal
policy of the MDP the loss solves. (1 - alpha) * tau is the same policy quoted in the unamplified
MDP's units, so dividing this q by it double-counts the factor and acts 10x sharper. The
observable side is gap(Q_tot) / gap(Q_env), which needs the Q_env head `temperature.tex`
proposes; it should settle near 10.
