"""The NumPy boundary. The rules themselves are covered by the crate's own `cargo test`."""

import ml_dtypes
import numpy as np
import numpy.typing as npt
import pytest

from expreplay import Column, EpisodeEnd, Kind, PrioTree, Span, Timeline, Uniform

NUM_ENVS = 3
CAPACITY = 16


def run(
    timeline: Timeline, steps: int, *, end_at: int | None = None
) -> list[npt.NDArray[np.int64]]:
    """Resets, then steps every environment, terminating environment 0 at step ``end_at``."""
    transitions = []
    timeline.reset()
    for step in range(steps):
        terminated = np.zeros(NUM_ENVS, np.bool_)
        terminated[0] = step == end_at
        obs, stepped = timeline.step(terminated, np.zeros(NUM_ENVS, np.bool_))
        transitions.append((obs - (0, 1))[stepped])
    return transitions


def test_step_hands_out_handles_and_marks_the_autoreset_call() -> None:
    timeline = Timeline(NUM_ENVS, CAPACITY)
    assert timeline.newest.tolist() == [-1, -1, -1]

    obs = timeline.reset()
    assert obs.dtype == np.int64
    assert obs.tolist() == [[0, 0], [1, 0], [2, 0]]

    transitions = run(timeline, 4, end_at=1)
    # Environment 0 spends one call on the next episode's first observation.
    assert [len(t) for t in transitions] == [3, 3, 2, 3]
    assert timeline.newest.tolist() == [4, 4, 4]
    assert (timeline.num_envs, timeline.env_capacity) == (NUM_ENVS, CAPACITY)


def test_rollout_pads_past_a_terminal_and_stops_the_stack_at_the_start() -> None:
    timeline = Timeline(NUM_ENVS, CAPACITY)
    run(timeline, 6, end_at=2)
    span = Span(3, stack=2, episode_end=EpisodeEnd.PAD)
    assert (span.length, span.stack, span.episode_end) == (3, 2, EpisodeEnd.PAD)

    # Environment 0: transitions 0..2, terminal at 2, final observation at 3, then 4 onwards.
    starts = np.array([[0, 1], [0, 4], [1, 0]])
    assert timeline.valid(starts, span).tolist() == [True, False, True]
    assert timeline.valid(starts.reshape(3, 1, 2), span).shape == (3, 1)

    rollout = timeline.rollout(starts[[0, 2]], span)
    assert rollout.steps[..., 1].tolist() == [[1, 2, 2], [0, 1, 2]]
    assert rollout.frames[..., 1].tolist() == [[0, 1, 2, 3, 3], [0, 0, 1, 2, 3]]
    assert (rollout.steps[..., 0] == [[0], [1]]).all()
    assert rollout.kinds.tolist() == [
        [Kind.STEP, Kind.TERMINAL, Kind.TERMINAL],
        [Kind.STEP, Kind.STEP, Kind.STEP],
    ]
    assert rollout.lengths.tolist() == [2, 3]

    nested = timeline.rollout(starts[[0, 2]].reshape(2, 1, 2), span)
    assert nested.steps.shape == (2, 1, 3, 2) and nested.kinds.shape == (2, 1, 3)
    assert (nested.frames == rollout.frames[:, None]).all()
    assert nested.lengths.tolist() == [[2], [3]]

    with pytest.raises(ValueError, match="cannot start a rollout"):
        timeline.rollout(starts, span)


def test_rollout_continues_through_an_episode_end() -> None:
    timeline = Timeline(NUM_ENVS, CAPACITY)
    run(timeline, 6, end_at=2)
    span = Span(4, episode_end=EpisodeEnd.CONTINUE)

    rollout = timeline.rollout(np.array([[0, 1]]), span)
    assert rollout.steps[0, :, 1].tolist() == [1, 2, 3, 4]
    assert rollout.kinds[0].tolist() == [
        Kind.STEP,
        Kind.TERMINAL,
        Kind.FINAL,
        Kind.STEP,
    ]
    assert rollout.frames[0, :, 1].tolist() == [1, 2, 3, 4, 5]
    # The newest four steps of every environment, as an on-policy rollout reads them.
    newest: npt.NDArray[np.int64] = np.stack(
        [np.arange(NUM_ENVS), timeline.newest - 4], axis=-1
    )
    assert timeline.valid(newest, span).all()
    assert not timeline.valid(newest + (0, 1), span).any()


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.float32, np.int16, ml_dtypes.bfloat16, "<f8"]
)
def test_column_round_trips_any_dtype(dtype: npt.DTypeLike) -> None:
    column = Column(NUM_ENVS, CAPACITY, dtype=dtype, shape=(2, 3))
    assert (column.dtype, column.shape) == (np.dtype(dtype), (2, 3))
    assert (column.num_envs, column.env_capacity) == (NUM_ENVS, CAPACITY)

    handles = np.array([[0, 0], [2, 0], [0, 1], [1, 0]])
    values = np.arange(24).reshape(4, 2, 3).astype(dtype)
    assert column.write(handles, values).tolist() == [True] * 4

    out = column.read(handles[::-1])
    assert out.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(out, values[::-1])

    # Any leading shape, and a values array that is not contiguous.
    grid = handles.reshape(2, 2, 2)
    assert column.write(
        grid, np.ascontiguousarray(values[::-1]).reshape(2, 2, 2, 3)[:, ::-1]
    ).all()
    np.testing.assert_array_equal(
        column.read(grid[:, ::-1]), values[::-1].reshape(2, 2, 2, 3)
    )


def test_column_reports_stale_writes_and_refuses_stale_reads() -> None:
    column = Column(1, 4, dtype=np.float32)
    for t in range(6):
        column.write(np.array([[0, t]]), np.array([t], np.float32))

    landed = column.write(np.array([[0, 1], [0, 5]]), np.array([9, 9], np.float32))
    assert landed.tolist() == [False, True]
    assert column.read(np.array([[0, 2], [0, 5]])).tolist() == [2, 9]

    with pytest.raises(IndexError, match="overwritten"):
        column.read(np.array([[0, 1]]))
    with pytest.raises(IndexError, match="not been written"):
        column.read(np.array([[0, 6]]))
    with pytest.raises(IndexError, match="out of range"):
        column.read(np.array([[1, 0]]))

    # A skipped timestep reads as zero rather than as the previous lap.
    column.write(np.array([[0, 8]]), np.array([8], np.float32))
    assert column.read(np.array([[0, 6], [0, 7], [0, 8]])).tolist() == [0, 0, 8]


def test_column_casts_nothing_and_checks_shapes() -> None:
    column = Column(NUM_ENVS, CAPACITY, dtype=np.uint8, shape=(2,))
    handles = np.array([[0, 0]])

    with pytest.raises(TypeError, match="uint8"):
        column.write(handles, np.zeros((1, 2), np.float32))
    with pytest.raises(TypeError, match="uint8"):
        column.write(handles, [[0, 0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        column.write(handles, np.zeros((1, 3), np.uint8))
    with pytest.raises(TypeError, match="int64"):
        column.read(handles.astype(np.int32))
    with pytest.raises(ValueError, match=r"\[\.\.\., 2\]"):
        column.read(np.zeros((1, 3), np.int64))
    with pytest.raises(ValueError, match="negative"):
        column.read(np.array([[0, -1]]))
    with pytest.raises(TypeError, match="objects"):
        Column(1, 4, dtype=object)
    with pytest.raises(MemoryError):
        Column(1 << 20, 1 << 20, dtype=np.float64, shape=(1 << 20,))

    assert column.read(np.zeros((0, 2), np.int64)).shape == (0, 2)


def test_prio_tree_draws_valid_transitions_by_weight() -> None:
    timeline = Timeline(NUM_ENVS, CAPACITY)
    tree = PrioTree(NUM_ENVS, CAPACITY, seed=0)
    for transitions in run(timeline, 8, end_at=3):
        tree.write(transitions, np.ones(len(transitions), np.float32))
    assert tree.total == 23.0
    assert (tree.num_envs, tree.env_capacity) == (NUM_ENVS, CAPACITY)

    span = Span(3, stack=4, episode_end=EpisodeEnd.PAD)
    heavy = np.array([[1, 2]])
    tree.write(heavy, np.array([40], np.float32))

    for stratified in (False, True):
        handles, weights = tree.sample(512, timeline, span, stratified=stratified)
        assert handles.shape == (512, 2) and weights.shape == (512,)
        assert timeline.valid(handles, span).all()
        np.testing.assert_array_equal(tree.read(handles), weights)
        share = (handles == heavy).all(axis=-1).mean()
        assert 0.6 < share < 0.9

    with pytest.raises(ValueError, match="not finite"):
        tree.write(heavy, np.array([np.nan], np.float32))
    with pytest.raises(TypeError, match="float32"):
        tree.write(heavy, np.array([1.0]))
    with pytest.raises(RuntimeError, match="every weight is zero"):
        PrioTree(NUM_ENVS, CAPACITY).sample(1, timeline, span)


def test_uniform_draws_valid_transitions_and_is_seeded() -> None:
    timeline = Timeline(NUM_ENVS, CAPACITY)
    span = Span(2, episode_end=EpisodeEnd.PAD)
    with pytest.raises(RuntimeError, match="gave up"):
        Uniform(seed=0).sample(1, timeline, span)

    run(timeline, 8, end_at=3)
    handles = Uniform(seed=0).sample(256, timeline, span)
    assert handles.shape == (256, 2)
    assert timeline.valid(handles, span).all()
    assert len(np.unique(handles, axis=0)) > 12
    np.testing.assert_array_equal(handles, Uniform(seed=0).sample(256, timeline, span))


def test_span_rejects_what_it_cannot_mean() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Span(0, episode_end=EpisodeEnd.PAD)
    with pytest.raises(TypeError, match="EpisodeEnd"):
        Span(1, episode_end=0)  # type: ignore[arg-type]
    assert repr(Span(3, stack=4, episode_end=EpisodeEnd.CONTINUE)) == (
        "Span(3, stack=4, episode_end=EpisodeEnd.CONTINUE)"
    )
