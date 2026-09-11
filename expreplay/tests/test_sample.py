"""End-to-end checks that need the built extension."""

import numpy as np
import pytest

from expreplay import ReplayBuffer

NUM_ENVS = 3
CAPACITY = 64
OBS_SHAPE = (4, 4)
STACK = 4


def build(**kwargs) -> ReplayBuffer:
    return ReplayBuffer(
        NUM_ENVS,
        CAPACITY,
        obs_shape=OBS_SHAPE,
        seed=0,
        **{"obs_dtype": np.uint8, "act_dtype": np.uint8, **kwargs},
    )


def frame(value: int) -> np.ndarray:
    """One frame per environment, filled with its own step number."""
    return np.full((NUM_ENVS, *OBS_SHAPE), value, dtype=np.uint8)


def stacked(value: int) -> np.ndarray:
    """The environment's view: `(num_envs, STACK, *OBS_SHAPE)`, newest last."""
    return np.stack([frame(max(0, value - STACK + 1 + i)) for i in range(STACK)], axis=1)


def play(rb: ReplayBuffer, steps: int, *, terminate_at: frozenset[int] = frozenset()) -> None:
    """Step the buffer, filling each frame with a counter and each reward with the same number.

    The counter jumps at an episode boundary rather than carrying straight on, so that a frame
    stack which wrongly reached across one shows up as a jump instead of blending in.
    """
    value, ended = 0, False
    for step in range(steps):
        terminated = step in terminate_at
        nxt = value + 100 if ended else value + 1
        rb.save_step(
            np.zeros(NUM_ENVS, dtype=np.uint8),
            np.full(NUM_ENVS, float(value), dtype=np.float32),
            np.full(NUM_ENVS, terminated),
            np.zeros(NUM_ENVS, dtype=bool),
            frame(nxt),
        )
        value, ended = nxt, terminated


# Every dtype the buffer stores, in the order `dyn_array.rs` lists them. These only reach
# Python -- `cargo test` builds no interpreter, so the boundary that converts them is the one
# thing the Rust tests cannot touch.
DTYPES = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
]


@pytest.mark.parametrize("dtype", DTYPES)
def test_every_supported_dtype_survives_the_round_trip(dtype) -> None:
    """Store it, draw it back, and check the value made it through unchanged."""
    rb = ReplayBuffer(
        NUM_ENVS,
        CAPACITY,
        obs_shape=(1,),
        obs_dtype=dtype,
        act_shape=(),
        act_dtype=dtype,
        obs_stack=STACK,
        seed=0,
    )
    assert rb.obs_dtype == np.dtype(dtype) and rb.act_dtype == np.dtype(dtype)

    rb.reset(np.zeros((NUM_ENVS, STACK, 1), dtype))
    for step in range(1, 40):
        rb.save_step(
            np.full(NUM_ENVS, step, dtype),
            np.full(NUM_ENVS, float(step), np.float32),
            np.zeros(NUM_ENVS, bool),
            np.zeros(NUM_ENVS, bool),
            np.full((NUM_ENVS, STACK, 1), step, dtype),
        )

    _i, _p, obs, act, _r, _t, next_obs = rb.sample(64)
    assert obs.dtype == np.dtype(dtype) and next_obs.dtype == np.dtype(dtype)
    assert act.dtype == np.dtype(dtype)

    # `play` above stores the step counter in both. The action recorded with a transition is
    # the one taken *from* its observation, so it is the counter of the frame after it.
    newest = obs[:, -1, 0].astype(np.int64)
    assert (act.astype(np.int64) == newest + 1).all()
    assert (next_obs[:, -1, 0].astype(np.int64) == newest + 1).all()


def test_an_unsupported_dtype_names_the_ones_that_would_have_worked() -> None:
    for bad in (np.complex64, np.bool_, "U4"):
        with pytest.raises(TypeError, match="Unsupported dtype"):
            build(obs_dtype=bad)

    # The message is built from the same list the buffer dispatches on, so it stays true.
    with pytest.raises(TypeError, match="uint8, uint16, .*float32, float64"):
        build(obs_dtype=np.complex64)


def test_a_mismatched_dtype_says_which_argument_and_what_it_wanted() -> None:
    """Nothing is cast on the way in, and the refusal has to be readable."""
    rb = ReplayBuffer(
        NUM_ENVS, CAPACITY, obs_shape=OBS_SHAPE, obs_dtype=np.uint8, act_dtype=np.int32
    )

    with pytest.raises(TypeError, match="obs must be a uint8 array, got a float32"):
        rb.reset(frame(0).astype(np.float32))

    rb.reset(frame(0))
    with pytest.raises(TypeError, match="actions must be a int32 array, got a uint8"):
        rb.save_step(
            np.zeros(NUM_ENVS, np.uint8),
            np.zeros(NUM_ENVS, np.float32),
            np.zeros(NUM_ENVS, bool),
            np.zeros(NUM_ENVS, bool),
            frame(1),
        )
    with pytest.raises(TypeError, match="next_obs must be a uint8 array, got a list"):
        rb.save_step(
            np.zeros(NUM_ENVS, np.int32),
            np.zeros(NUM_ENVS, np.float32),
            np.zeros(NUM_ENVS, bool),
            np.zeros(NUM_ENVS, bool),
            frame(1).tolist(),
        )


def test_float16_takes_a_real_array_and_nothing_else() -> None:
    """The one dtype with no element-by-element fallback: pyo3 cannot build an f16 from a float."""
    rb = ReplayBuffer(
        NUM_ENVS, CAPACITY, obs_shape=(2,), obs_dtype=np.float16, act_dtype=np.uint8
    )
    rb.reset(np.zeros((NUM_ENVS, 2), np.float16))

    with pytest.raises(TypeError, match="obs must be a float16 array, got a list"):
        rb.reset([[0.0, 0.0]] * NUM_ENVS)

    # Not a cast either -- a float32 array is refused the same way every other dtype is.
    with pytest.raises(TypeError, match="obs must be a float16 array"):
        rb.reset(np.zeros((NUM_ENVS, 2), np.float32))


def test_sample_shapes_and_dtypes() -> None:
    rb = build(obs_stack=STACK)
    rb.reset(frame(0))
    play(rb, 40)

    indices, prios, obs, act, rewards, terminals, next_obs = rb.sample(16)

    assert indices.shape == (16,) and indices.dtype == np.uint32
    assert prios.shape == (16,) and prios.dtype == np.float32
    assert obs.shape == (16, STACK, *OBS_SHAPE) and obs.dtype == np.uint8
    assert next_obs.shape == obs.shape
    assert act.shape == (16,) and act.dtype == np.uint8
    assert rewards.shape == (16,) and rewards.dtype == np.float32
    assert terminals.shape == (16,) and terminals.dtype == np.bool_


def test_sample_returns_consecutive_frames_and_the_right_successor() -> None:
    rb = build(obs_stack=STACK)
    rb.reset(frame(0))
    play(rb, 40)

    _indices, _prios, obs, _act, rewards, terminals, next_obs = rb.sample(64)

    # With no terminations the counter is just the step number, so a stack reads as consecutive
    # integers ending on the newest.
    newest = obs[:, -1].reshape(64, -1)[:, 0].astype(int)
    for offset in range(STACK):
        assert (obs[:, STACK - 1 - offset].reshape(64, -1)[:, 0] == newest - offset).all()

    # One-step returns: the reward stored with a transition is its own step number, and its
    # successor observation is the frame after it.
    assert (rewards == newest).all()
    assert (next_obs[:, -1].reshape(64, -1)[:, 0] == newest + 1).all()
    assert not terminals.any()


def test_step_accepts_a_stacked_observation() -> None:
    stacked_rb, single_rb = build(obs_stack=STACK), build(obs_stack=STACK)
    stacked_rb.reset(stacked(0))
    single_rb.reset(frame(0))

    for step in range(40):
        args = (
            np.zeros(NUM_ENVS, dtype=np.uint8),
            np.full(NUM_ENVS, float(step), dtype=np.float32),
            np.zeros(NUM_ENVS, dtype=bool),
            np.zeros(NUM_ENVS, dtype=bool),
        )
        stacked_rb.save_step(*args, stacked(step + 1))
        single_rb.save_step(*args, frame(step + 1))

    # The same seed and the same stored frames, so the same batch.
    for a, b in zip(stacked_rb.sample(16), single_rb.sample(16)):
        assert (np.asarray(a) == np.asarray(b)).all()


def test_frame_stacks_do_not_cross_an_episode_boundary() -> None:
    rb = build(obs_stack=STACK)
    rb.reset(frame(0))
    play(rb, 40, terminate_at=frozenset({20}))

    _indices, _prios, obs, *_ = rb.sample(256)

    # Every stack is some number of repeats of its oldest frame followed by consecutive ones: the
    # walk only ever pads, and only at the start of an episode. A stack that crossed the boundary
    # would show the counter's jump of 100.
    flat = obs.reshape(256, STACK, -1)[:, :, 0].astype(int)
    for row in flat:
        diffs = np.diff(row)
        consecutive = int(np.argmax(diffs == 1)) if (diffs == 1).any() else len(diffs)
        assert (diffs[:consecutive] == 0).all(), f"padding is not a prefix: {row}"
        assert (diffs[consecutive:] == 1).all(), f"stack crosses a boundary: {row}"


def test_n_step_returns_and_terminals() -> None:
    rb = build(obs_stack=STACK)
    rb.reset(frame(0))
    play(rb, 40, terminate_at=frozenset({20}))

    _i, _prios, obs, _a, rewards, terminals, _n = rb.sample(256, n_steps=3, discount=0.5)

    # `play` sets each reward to the same counter as the frame, so a full three-step return from
    # a transition whose newest frame is `v` is `v + (v+1)/2 + (v+2)/4`. Rollouts a terminal cut
    # short are exactly the ones flagged terminal, so filtering those leaves only full ones.
    start = obs[:, -1].reshape(256, -1)[:, 0].astype(np.float32)
    full = start + (start + 1) / 2 + (start + 2) / 4
    assert terminals.any(), "the terminated transition should turn up in 256 draws"
    assert np.allclose(rewards[~terminals], full[~terminals])


def test_a_batch_larger_than_the_buffer_still_draws() -> None:
    """Three samplable transitions per environment, and a batch two orders of magnitude larger."""
    rb = build(obs_stack=STACK)
    rb.reset(frame(0))
    play(rb, 6)

    indices, *_ = rb.sample(256)
    assert indices.shape == (256,)


def test_stratified_needs_priorities_to_stratify() -> None:
    with pytest.raises(ValueError, match="stratified requires use_prios"):
        build(stratified=True)

    rb = build()
    assert not rb.stratified
    with pytest.raises(ValueError, match="stratified requires use_prios"):
        rb.stratified = True
    assert not rb.stratified

    # With priorities it is the default, and settable either way.
    prioritised = build(use_prios=True)
    assert prioritised.stratified
    prioritised.stratified = False
    assert not prioritised.stratified
    prioritised.stratified = True
    assert prioritised.stratified


def test_sample_needs_something_to_draw_from() -> None:
    rb = build(obs_stack=STACK)
    rb.reset(frame(0))
    play(rb, 2)

    with pytest.raises(ValueError, match="enough transitions"):
        rb.sample(4)


@pytest.mark.parametrize("stratified", [True, False])
def test_prioritised_sampling_follows_update_prios(stratified: bool) -> None:
    rb = build(use_prios=True, stratified=stratified)
    rb.reset(frame(0))
    play(rb, 40)

    indices, _prios, *_ = rb.sample(64)

    # Every written transition starts at `max_prio`, so zeroing only the drawn ones would leave
    # the rest competing. Zero the whole buffer, then give one known-samplable transition all of
    # the mass: it has to win every draw.
    everything = np.arange(rb.total_capacity, dtype=np.uint32)
    rb.update_prios(everything, np.zeros(len(everything), dtype=np.float32))
    rb.update_prios(indices[:1], np.full(1, 7.0, dtype=np.float32))

    drawn, prios, *_ = rb.sample(32)
    assert (drawn == indices[0]).all()

    # The priority comes back exactly as it was stored, not divided through by the total -- which
    # here is the same number, and is what turns it into the probability of one that it is.
    assert np.allclose(prios, 7.0)
    assert rb.sum_prios == pytest.approx(7.0)
    assert prios[0] / rb.sum_prios == pytest.approx(1.0)


def test_sample_after_the_buffer_wraps() -> None:
    """The steady state of a real run: every age is modular, and none may reach the write head."""
    rb = build(obs_stack=STACK)
    rb.reset(frame(0))
    play(rb, 150, terminate_at=frozenset({37}))
    assert len(rb) == rb.total_capacity, "150 steps should have wrapped 64 slots"

    _i, prios, obs, _a, rewards, terminals, next_obs = rb.sample(1024, n_steps=3, discount=0.5)

    # Both stacks stay well formed, and `next_obs` is the full three steps ahead: a wrapped age
    # that ran onto the head would show up as a stack out of order or a successor out of place.
    newest = obs[:, -1].reshape(1024, -1)[:, 0].astype(int)
    assert (next_obs[:, -1].reshape(1024, -1)[:, 0] == newest + 3).all()
    assert not terminals.any(), "the one terminal is 113 steps back, so it has been overwritten"

    start = newest.astype(np.float32)
    assert np.allclose(rewards, start + (start + 1) / 2 + (start + 2) / 4)

    # A uniform buffer reports every draw at 1.0, so the weights it implies are all one.
    assert (prios == 1.0).all()


def test_sample_survives_priorities_that_have_collapsed_to_nothing() -> None:
    """A total this small leaves no room between the slices of a stratified draw."""
    rb = build(obs_stack=STACK, use_prios=True, stratified=True)
    rb.reset(frame(0))
    play(rb, 40)

    indices, *_ = rb.sample(32)
    tiny = np.float32(np.finfo(np.float32).smallest_subnormal)
    rb.update_prios(np.arange(rb.total_capacity, dtype=np.uint32), np.zeros(rb.total_capacity, np.float32))
    rb.update_prios(indices[:1], np.full(1, tiny, dtype=np.float32))

    drawn, prios, *_ = rb.sample(32)
    assert (drawn == indices[0]).all()
    assert (prios == tiny).all()


def test_parameters_are_properties() -> None:
    rb = build(obs_stack=STACK, use_prios=True, max_prio_decay=0.5, stratified=False)

    assert rb.num_envs == NUM_ENVS
    assert rb.env_capacity == CAPACITY
    assert rb.total_capacity == NUM_ENVS * CAPACITY
    assert rb.obs_shape == OBS_SHAPE and rb.obs_dtype == np.uint8
    assert rb.act_shape == () and rb.act_dtype == np.uint8
    assert rb.obs_stack == STACK
    assert rb.use_prios and rb.sum_prios == 0.0
    assert build().obs_stack is None and not build().use_prios

    # The two that a schedule may want to move mid-run.
    assert rb.max_prio_decay == 0.5 and not rb.stratified
    rb.max_prio_decay, rb.stratified = 0.9, True
    assert rb.max_prio_decay == pytest.approx(0.9) and rb.stratified

    with pytest.raises(ValueError, match="max_prio_decay"):
        rb.max_prio_decay = 0.0

    # Read-only, and `max_prio` tracks the decay it is given.
    with pytest.raises(AttributeError):
        rb.num_envs = 1  # type: ignore[misc]

    assert rb.max_prio == 1.0
    rb.reset(frame(0))
    play(rb, 8)
    indices, *_ = rb.sample(4)
    rb.update_prios(indices[:1], np.full(1, 0.25, dtype=np.float32))
    assert rb.max_prio == pytest.approx(0.9)
