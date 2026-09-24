# Performance

Throughput measurements of the training loop, oldest first. Hardware throughout: RTX 3080 +
24-thread host, 64 envs, batch 256. Individual benchmark runs are in `perf/RUNS.md`.

Read the rate off TensorBoard's `run/env_steps_per_second`, not off a trace (see the profiler
section). Compare rates only at equal `LOG_FREQ`: the window is the averaging interval, so a
longer one hides brief dips.

## The profiling chain, 2026-09-13

1:1 act/train, steady state, each step of the chain over two or three runs:

| env steps/s | change |
|---|---|
| 1034 | where the profiling started |
| 1112 | `--xla_gpu_force_conv_nhwc`, since removed: see `IMPALA_DTYPE` |
| 1237-1261 | the priority write-back queued, `PRIO_QUEUE_SLACK` |
| 2152-2153 | the trunk's convolutions in bfloat16, `IMPALA_DTYPE` |

That is 61.9 -> 29.7 ms an iteration, ~8.6k ALE frames/s, ~6.5 h for 50M env steps / 200M
frames.

Held over a full run: 2089 env steps/s median across 6.9 h and 824k gradient steps, flat from the
first decile to the last. Moving the per-parameter scales to `SCALE_LOG_FREQ` is worth the last
5.9% of that (they used to be reductions inside every gradient step), and it also lifts the slow
tail, p5 1906 against 1694 over windows of equal length.

Logging cadence: `_train_step` returns thirty-one series over arrays the step already holds, so
only their transfer is worth thinning. The ~130 per-parameter scales move over a whole run and
say nothing new at a fine cadence. At `TRAIN_LOG_FREQ` 100 and `SCALE_LOG_FREQ` 2500 a
200M-frame run writes ~30 MB of event file, where logging everything at 25 wrote 384.

## jax.profiler trace, 2026-09-13

Gradient steps 20..80 (`--profile 20:60`), per iteration. The kernel durations are the device's
own and stand as they are. Every host span also carries CUPTI's per-launch cost, which is now
most of what the trace measures: the trace reports a 54.6 ms iteration against the 29.7 ms the
rate scalar sees. Read a host span as its share of 54.6 ms, never against the real wall.

| span | time | note |
|---|---|---|
| device kernels | 23.0 ms | 1153 launches an iteration, median 1.3 us; 77% of the real 29.7 |
| learn_step | 41.6 ms | host, of 54.6; almost all of it inside the executable call |
| env.step | 2.9 ms | host, ALE; it launches nothing, so the inflation misses it |
| observe | 0.3 ms | host: 0.22 ms in the buffer's save_step, 0.04 in update_prios |
| act transfer | 0.1 ms | host; the overlap works, this one is free |

## Per-part timing, 2026-09-18

Same host, `LAYER_NORM` on, evaluation and checkpointing off, by `perf/bench_loop.py`: 2253 env
steps/s, an iteration of 28.4 ms. Each part is timed alone, the device under saturation and the
host after a barrier, so both columns are what each part really costs:

| part | device | host |
|---|---|---|
| `_train_step` | 25-26 ms | 12-16 ms |
| `_act`, 64 envs | 0.9 ms | 3.9 ms |
| `sync_qnet`, before it stopped being jitted | 0 | 6.7 ms |
| `buf.sample` | | 1.0-2.0 ms, flat in how full the buffer is |
| the batch's h2d | | 1.2 ms, obs and next_obs together |
| `env.step`, 64 envs | | 1.6 ms |
| `save_step` | | 0.07 ms; update_prios 0.006 |

The device takes 26.2 ms of the 28.4 and the host, at ~22 ms, has slack. So every host saving
measured zero on the wall, and only device work is worth cutting.

`_train_step`'s device time is its trunk passes and nothing else: 4.4 ms for the Munchausen pass
over obs, 4.4 ms for the target on next_obs, and 16.7 ms for the online forward and backward.
They sum to the whole within 0.5%. The trunk holds ~30 TFLOP/s of bfloat16, about half of what
this card does dense, so there is no large kernel-level win left under it either.

`sync_qnet` unjitted drops 6.7 ms of host a sync, all of it nnx graph traversal for a call that
did nothing on the device. It buys no wall time: 2253 against 2247 env steps/s.

## Munchausen off the online pass

Reading the Munchausen log-policy off the online pass removes the first of those three trunk
passes, so the table above describes a three-pass gradient step. The same harness on Zaxxon,
the two runs back to back, 280 s each: 2304 -> 2614 env steps/s, +13.5%, 27.8 -> 24.5 ms an
iteration. The slower run's p90 is below the faster run's p10.

## After the third pass went

The loop is still device-bound, but the margin is narrow enough that the host is worth cutting
too. Measured with `perf/split_train_step.py`, three runs each, plus a `BENCH_RR=0/64` run for
the host cost left when no gradient step is taken at all:

| part | measurement |
|---|---|
| `_train_step` | 22.0 ms device (21.8-22.8), against 26.4 with the third pass (26.4-26.5) |
| | 14.1 ms host to dispatch, 7.0 with its modules bound |
| `_act`, 64 envs | 1.8 ms device, 1.7 ms host, bound |
| loop, no grad step | 12154 env steps/s, 5.3 ms an iteration: env.step, save_step and _act |

So there is ~14 ms of host against ~24 ms of device in a 24.5 ms iteration, and the GPU sets the
rate.

Binding `_train_step`'s modules is the 14.1 -> 7.0 ms, and what is left is still nnx's.
`nnx.split` of the four alone is 5.1 ms for 297 array leaves, against a 1.1 ms `jax.jit` dispatch
floor for the same arrays. That floor is what a pure-JAX kernel over a state pytree would cost,
and such a kernel is the only way to recover the remaining 6 ms. The binding buys no wall time:
2633 env steps/s against 2457 in one pair, with an earlier run of the same unbound code at 2614,
so run-to-run drift swamps it. It is kept for the CPU saving, which is real and matters to the
rest of the machine: 3.77 cores against 3.87 over an equal wall, 92 against 101 ms of CPU a
gradient step.

## Replay ratio

Gradient steps an iteration against env steps/s, everything else held (`perf/sweep.sh`):

| grad steps / iteration | env steps/s | 200M-frame budget |
|---|---|---|
| 1 | 2253 | 6.2 h |
| 2 | 1150 | 12.1 h |
| 4 | 602 | 23.1 h |
| 8 | 294 | 47.2 h |

Linear to within 5%, and that 5% is the per-iteration cost amortising rather than slack being
taken up: a gradient step costs its own 25 ms of device wherever it lands.

## Evaluation's cost

The evaluator plays continuously, so its cost is paid on every game. Under the old snapshot
evaluator, one 8-episode evaluation on Phoenix outlasted 6250 gradient steps, so both
`EVAL_FREQ` windows inside a 280 s run were skipped.

What evaluation costs the loop is `_act`'s host time: Python holding the GIL against the loop's
own dispatch, not the two contending for the device. Binding the acting net took 60% of that
back:

| | before | after | Mann-Whitney |
|---|---|---|---|
| evaluation on | 2133 | 2206 | p = 5e-10 over ~70 windows |
| evaluation off | 2257 | 2255 | p = 0.72 |

That is a 5.5% tax down to 2.2%, and no change in the training loop itself. Host savings show
up only where a thread is host-bound, and the loop is not.

## Where the device time goes now

Convolutions still take the largest share of the device, at ~18%, but they no longer dominate
it. What is left of the idle time is the launch-bound tail: 81% of launches are under 5 us and
account for 6% of device time. Shrinking that means fewer kernels, not faster ones, and nothing
below has found a way to launch fewer.

## Measured and rejected

| tried | result |
|---|---|
| XLA command buffers | no change, at any `--xla_gpu_enable_command_buffer` setting |
| TF32 matmul precision | `JAX_DEFAULT_MATMUL_PRECISION` moves dots, not cuDNN's convs, which pick their own math type; no change either way |
| dropping the diagnostics | 0.8 ms/step of 59, so utils' "costs nothing" claim holds |
| dropping SpectralNorm | 2.0 ms/step of 59; not where the time goes |
| a queue of 5 batches | 1280 against 1237 and 1261 at a queue of 3, inside the run-to-run spread, so `PRIO_QUEUE_SLACK` keeps the shallower queue and its smaller stale window |
| `--xla_gpu_force_conv_nhwc` | worth 8% while the trunk was float32, nothing once it is bfloat16: XLA already lays 16-bit convolutions out NHWC |

## Figures moved out of the code, 2026-09-24

- **`_scale_stats`:** dispatched one at a time, its ~130 reductions over every parameter and
  Adam moment cost 48 ms of host, against 1.4 ms jitted. `graph=False` avoids graph mode writing
  the whole 21 MB of parameters and moments back out as fresh buffers on every call.
- **Binding `_act`** (`nnx.cached_partial`): 3.8 -> 1.6 ms of host per act.
- **Binding `_train_step`:** 14.1 -> 7.0 ms of host per call, all of it nnx's graph walk. Checked
  safe three ways:
  - `opt.update` leaves the graph structure alone, so the cache cannot go stale across a step.
  - `nnx.capture`'s sowing still reaches the bound modules, and the plasticity diagnostics track
    the unbound kernel's to 4e-4.
  - None of it survives the call: `nnx.state(qnet)` is the same 82 leaves a checkpoint held
    before.
- **The unjitted `sync_qnet` alias:** checked against the jitted copy it replaced, on parameters,
  the actions the two nets draw, the flags, and SpectralNorm's carried `u` under acting.
- **Priority write-back:** reading back the gradient step just dispatched costs the host the
  whole of that step, 12.9 ms an iteration, against 0.22 ms in `save_step`. The delayed
  write-back's cost in exactness is small:
  - Slots recycled between a batch's draw and its write-back are bounded at
    2 * PRIO_QUEUE_SLACK = four of an environment's 16384, so 0.06 draws of a 256 batch.
  - Transitions redrawn at their stale priority before the write-back lands come to
    256 * 256 / 1048576 = 0.06 of a batch per following draw, 0.13 over the two in flight.
- **`effective_rank`:** computed through the [batch, batch] Gram matrix, ~300 MFLOP against
  train_step's ~598 GFLOP, so no measurable cost.
- **The snapshot evaluator's skips:** scoring a snapshot meant a common reset ended by the slowest
  of `EVAL_NUM_ENVS` episodes, so each evaluation cost the maximum of that many episode lengths.
  On Phoenix that is near ALE's 108k-frame truncation: 6 of every 7 evaluations were skipped
  because the previous one was still running, against none on Qbert, and the episodes the fast
  envs finished in the meantime were thrown away. This is what moved the evaluator to a
  continuous fleet.
