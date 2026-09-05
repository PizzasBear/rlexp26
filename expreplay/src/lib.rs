//! A replay buffer for vectorised environments, with optional prioritised sampling.
//!
//! # Storage layout
//!
//! Every array is indexed `[env, slot, ..]`, where each of the `num_envs` environments owns an
//! independent circular buffer of `env_capacity` slots. Transitions are addressed from Python by
//! the flat index `env * env_capacity + slot`, which is what [`ReplayBuffer::sample`] returns and
//! what [`ReplayBuffer::update_prios`] expects back. That index is a `u32`, so the constructor
//! rejects a `num_envs * env_capacity` that would not fit in one.
//!
//! Successor observations are not stored. The transition at slot `i` reads its `next_obs` from
//! slot `i + 1` of the same environment, and frame stacks read backwards from `i` the same way.
//! This halves the memory an Atari-sized buffer needs, at the cost of making a window of slots
//! around each environment's write head unsamplable: the head itself has no successor yet, and
//! `obs_stack` and `n_steps` widen that window in either direction.
//!
//! # Episode boundaries
//!
//! Boundaries are inferred from the `terminated` and `truncated` flags of each
//! [`ReplayBuffer::save_step`], under Gymnasium's default `AutoresetMode.NEXT_STEP`: the step that
//! ends an episode returns that episode's *final* observation, and the call after it returns the
//! new episode's first observation together with an action and a reward that the environment
//! ignored. This buffer follows the same shape. The terminating step stores its transition and
//! parks the final observation in the slot after it -- that slot never gets an action, so it is
//! never sampled -- and the autoreset call that follows completes no transition at all; its
//! placeholder action and reward are overwritten in place by the next real step.
//!
//! [`ReplayBuffer::reset`] is for the opening observation and for abandoning an episode by hand.
//! It is not needed at autoreset boundaries, and calling it there is harmless.
//!
//! # Frame stacks
//!
//! `obs_stack` is a property of the model rather than of a draw, so it is fixed at construction:
//! [`ReplayBuffer::sample`] returns that many consecutive frames per observation, on an axis just
//! after the batch. Only one frame per slot is ever stored -- the rest of a stack is already in
//! the slots before it -- so [`ReplayBuffer::save_step`] takes either a single new frame or the
//! environment's whole stacked observation, `(num_envs, obs_stack, *obs_shape)`, and keeps only
//! the newest frame of it. See [`obs_frame`]. `n_steps`, by contrast, is a per-draw argument,
//! since it is the kind of thing a schedule moves during training.
//!
//! Frames lead rather than trail so that each one is a contiguous run of bytes on both paths:
//! `save_step` picks its frame out with a memcpy per environment instead of a stride-`obs_stack` byte
//! scatter, and `sample` builds a stack as `obs_stack` back-to-back memcpys. A model wanting
//! frames trailing should transpose on the accelerator, where that move is cheap and is usually
//! folded into the first layer.
//!
//! # Priorities
//!
//! Sampling is proportional to the values most recently passed to [`ReplayBuffer::update_prios`],
//! and no exponent is applied to them on the way in. A caller wanting the usual `p ** alpha`
//! weighting therefore applies `alpha` before calling. Note that `alpha` genuinely cannot be
//! applied after the fact: the tree samples in proportion to whatever it stores, so raising the
//! priorities to a power afterwards would not change the distribution that produced the batch.
//!
//! [`ReplayBuffer::sample`] hands those stored priorities straight back rather than normalising
//! them. Which denominator an importance-sampling weight should use is the caller's decision, not
//! this buffer's -- [`sum_prios`](ReplayBuffer::sum_prios) gives the sampling probability,
//! [`max_prio`](ReplayBuffer::max_prio) a weight scaled against the whole buffer rather than
//! against one batch -- and a normalised number cannot be taken back apart. That keeps `beta` on
//! the caller's side along with `alpha`.

use std::num::NonZero;

use bytemuck::{Zeroable, allocation};
use ndarray::{par_azip, prelude::*};
use numpy::{
    PyArray1, PyArrayDescr, PyArrayDescrMethods, PyArrayLike1, PyArrayLikeDyn, PyArrayMethods,
};
use pyo3::{
    exceptions::{
        PyIndexError, PyMemoryError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError,
    },
    prelude::*,
    types::PyTuple,
};
use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
use thiserror::Error;

mod prio_tree;
mod sampling;

use prio_tree::PrioTree;

#[derive(Debug, Error)]
#[error("Allocation failed")]
struct AllocationError(());

impl AllocationError {
    const fn new() -> Self {
        Self(())
    }
}

impl From<AllocationError> for PyErr {
    fn from(value: AllocationError) -> Self {
        PyMemoryError::new_err(value.to_string())
    }
}

/// Allocates `len` zeroed `T`s, returning an error instead of aborting when that fails.
///
/// The usual ways to get a zeroed buffer -- `vec![0; n]`, `Array::zeros` -- route allocation
/// failure through [`handle_alloc_error`], which *aborts* the process rather than unwinding. That
/// is the wrong behaviour here: the observation array is routinely tens of gigabytes, so "you
/// asked for more than fits" is an ordinary, recoverable mistake that should reach Python as a
/// `MemoryError` instead of killing the interpreter. [`bytemuck`] supplies the fallible primitive
/// and the [`Zeroable`] bound that makes it sound.
///
/// [`allocation::try_zeroed_vec`] goes through `alloc_zeroed`, for the same reason `vec![0; n]`
/// does: it lets the allocator ask the OS for fresh pages, which are already zero and can be
/// mapped lazily. Building the buffer with `try_reserve` + `resize` would also be fallible, but
/// would memset every byte up front -- for a multi-gigabyte observation array that means paying
/// for the whole thing at construction rather than as it fills.
///
/// Note that on Linux under the default overcommit policy a very large request will usually
/// *succeed* here and only fail later, when the pages are first touched, at which point the OOM
/// killer decides what happens rather than this function. The check still earns its place: it is
/// the difference between a clean `MemoryError` and a hard abort on platforms that do not
/// overcommit, when overcommit is disabled, and when the request exhausts the address space
/// outright.
///
/// [`handle_alloc_error`]: std::alloc::handle_alloc_error
fn try_zeroed_vec<T: Zeroable>(len: usize) -> Result<Vec<T>, AllocationError> {
    allocation::try_zeroed_vec(len).map_err(|()| AllocationError::new())
}

impl From<prio_tree::UpdateError> for PyErr {
    fn from(value: prio_tree::UpdateError) -> Self {
        match value {
            prio_tree::UpdateError::InvalidPrio(_) => PyValueError::new_err(value.to_string()),
            prio_tree::UpdateError::IndexError => PyIndexError::new_err(value.to_string()),
        }
    }
}

impl From<sampling::DrawError> for PyErr {
    fn from(value: sampling::DrawError) -> Self {
        match value {
            // The caller asked for a draw the buffer cannot serve.
            sampling::DrawError::TooSmall | sampling::DrawError::NoPriorities => {
                PyValueError::new_err(value.to_string())
            }
            // Nothing the caller passed is wrong; the buffer's own state is degenerate.
            sampling::DrawError::Exhausted => PyRuntimeError::new_err(value.to_string()),
        }
    }
}

impl From<prio_tree::SampleError> for PyErr {
    fn from(value: prio_tree::SampleError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DType {
    F32,
    U8,
}

impl DType {
    fn from_np<'py>(dtype: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = dtype.py();

        let descr = PyArrayDescr::new(py, dtype)?;

        if descr.is_equiv_to(&numpy::dtype::<f32>(py)) {
            Ok(Self::F32)
        } else if descr.is_equiv_to(&numpy::dtype::<u8>(py)) {
            Ok(Self::U8)
        } else {
            Err(PyTypeError::new_err(format!(
                "Unsupported dtype {descr}, ReplayBuffer only supports float32 and uint8"
            )))
        }
    }

    fn np<'py>(self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match self {
            Self::U8 => numpy::dtype::<u8>(py),
            Self::F32 => numpy::dtype::<f32>(py),
        }
    }
}

/// One of the two dtypes the buffer stores, owned.
enum DynArray {
    U8(ArrayD<u8>),
    F32(ArrayD<f32>),
}

/// The borrowed counterpart: a batch handed over by Python, narrowed on its way into a slot.
enum DynArrayView<'a> {
    U8(ArrayViewD<'a, u8>),
    F32(ArrayViewD<'a, f32>),
}

impl DynArray {
    #[inline]
    const fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::F32(_) => DType::F32,
        }
    }

    #[inline]
    fn shape(&self) -> &[usize] {
        match self {
            Self::U8(array) => array.shape(),
            Self::F32(array) => array.shape(),
        }
    }

    /// Allocates a zeroed array of `shape`, failing rather than aborting if it does not fit.
    ///
    /// This is by far the largest allocation the buffer makes -- for Atari-sized observations it
    /// is three orders of magnitude bigger than everything else here put together -- so it is the
    /// one that most needs to surface as a `MemoryError` instead of taking the interpreter with
    /// it. See [`try_zeroed_vec`].
    fn try_zeros(shape: &[usize], dtype: DType) -> Result<Self, AllocationError> {
        fn zeros<T: Zeroable>(shape: &[usize]) -> Result<ArrayD<T>, AllocationError> {
            let len = shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                .ok_or_else(AllocationError::new)?;

            Ok(ArrayD::from_shape_vec(IxDyn(shape), try_zeroed_vec(len)?)
                .expect("`len` is the product of the shape, so the array cannot reject it"))
        }

        Ok(match dtype {
            DType::F32 => Self::F32(zeros(shape)?),
            DType::U8 => Self::U8(zeros(shape)?),
        })
    }
}

impl<'a> DynArrayView<'a> {
    #[inline]
    fn shape(&self) -> &[usize] {
        match self {
            Self::U8(array) => array.shape(),
            Self::F32(array) => array.shape(),
        }
    }

    /// Narrows to one index along `axis`, dropping that axis from the shape.
    fn index_axis_move(self, axis: Axis, index: usize) -> Self {
        match self {
            Self::U8(array) => Self::U8(array.index_axis_move(axis, index)),
            Self::F32(array) => Self::F32(array.index_axis_move(axis, index)),
        }
    }
}

/// [`write_batch`] over a pair of arrays whose dtype is only known at runtime.
fn dyn_write_batch(dst: &mut DynArray, slots: &[u32], src: &DynArrayView<'_>) {
    match (dst, src) {
        (DynArray::U8(dst), DynArrayView::U8(src)) => write_batch(dst, slots, src),
        (DynArray::F32(dst), DynArrayView::F32(src)) => write_batch(dst, slots, src),
        _ => unreachable!("the caller extracts `src` with `dst`'s own dtype"),
    }
}

/// Gathers `slots` into a fresh NumPy array of `shape` in `src`'s dtype.
///
/// The gather itself sees one row per slot, so `shape` is first flattened to
/// `[slots.len(), ..tail]` -- which folds a stack axis into the batch, and leaves an unstacked
/// `shape` alone. Reshaping like that is sound because the array was allocated here and is
/// contiguous.
fn dyn_gather<'py>(
    py: Python<'py>,
    src: &DynArray,
    shape: &[usize],
    slots: &[u32],
    capacity: usize,
) -> Bound<'py, PyAny> {
    fn gather<'py, T>(
        py: Python<'py>,
        src: &ArrayD<T>,
        shape: &[usize],
        slots: &[u32],
        capacity: usize,
    ) -> Bound<'py, PyAny>
    where
        T: numpy::Element + Clone + Send + Sync + 'static,
    {
        // `src` is `[env, slot, ..tail]`, so everything past the slot axis is one stored row.
        let mut rows = vec![slots.len()];
        rows.extend_from_slice(&src.shape()[2..]);

        let out = numpy::PyArrayDyn::<T>::zeros(py, shape, false);
        {
            let mut guard = out.readwrite();
            let mut view = guard
                .as_array_mut()
                .into_shape_with_order(IxDyn(&rows))
                .expect("an array allocated here is contiguous");

            sampling::gather_batch(&mut view, src, slots, capacity);
        }

        out.into_any()
    }

    match src {
        DynArray::U8(src) => gather(py, src, shape, slots, capacity),
        DynArray::F32(src) => gather(py, src, shape, slots, capacity),
    }
}

/// Number of elements past which a per-row copy is handed to rayon.
///
/// A batch of stacked Atari observations is a few hundred kilobytes and is worth spreading over
/// the pool; a batch of one reward or one action per environment is a few hundred *bytes*, where
/// a fork-join costs far more than the copy itself. The exact cut-off does not matter much --
/// anything near it is cheap whichever way it goes.
pub(crate) const PAR_COPY_MIN_ELEMS: usize = 1 << 14;

/// Copies row `env` of `src` into `dst[env, slots[env]]`, for every environment at once.
///
/// `dst` is a stored `[env, slot, ..]` array and `src` the per-environment batch that goes into it.
fn write_batch<T, D>(dst: &mut ArrayRef<T, D>, slots: &[u32], src: &ArrayRef<T, D::Smaller>)
where
    T: Copy + Send + Sync + 'static,
    D: Dimension + ndarray::RemoveAxis,
    D::Smaller: ndarray::RemoveAxis,
{
    if src.len() < PAR_COPY_MIN_ELEMS {
        azip!((mut dst in dst.outer_iter_mut(), &slot in slots, src in src.outer_iter()) {
            dst.index_axis_mut(Axis(0), slot as _).assign(&src);
        });
    } else {
        par_azip!((mut dst in dst.outer_iter_mut(), &slot in slots, src in src.outer_iter()) {
            dst.index_axis_mut(Axis(0), slot as _).assign(&src);
        });
    }
}

enum DynPyArrayLike<'py> {
    U8(PyArrayLikeDyn<'py, u8>),
    F32(PyArrayLikeDyn<'py, f32>),
}

impl<'py> DynPyArrayLike<'py> {
    fn extract(dtype: DType, obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(match dtype {
            DType::U8 => Self::U8(obj.extract()?),
            DType::F32 => Self::F32(obj.extract()?),
        })
    }

    fn as_array(&self) -> DynArrayView<'_> {
        match self {
            Self::U8(a) => DynArrayView::U8(a.as_array()),
            Self::F32(a) => DynArrayView::F32(a.as_array()),
        }
    }
}

/// How an episode continues past a slot, kept per slot so that sampling can tell a real episode
/// end from a time limit and can avoid stacking frames across a boundary.
///
/// Between calls an environment's write head always sits on a [`Reset`](Self::Reset) or a
/// [`Final`](Self::Final) slot; the remaining variants mark slots whose transition is complete,
/// and only those are ever samplable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum SlotType {
    /// Empty. The previous episode ended here and no observation has been written yet.
    Reset = 0,
    /// A complete transition whose episode carries on into the next slot.
    Normal = 1,
    /// Holds an observation, but not yet the action and reward taken from it.
    Final = 2,
    /// A complete transition that ended its episode, so nothing is bootstrapped past it.
    Terminal = 3,
    /// A complete transition cut short by a time limit or a mid-run [`ReplayBuffer::reset`]. Its
    /// successor observation is stored, so it is still worth bootstrapping from.
    Truncated = 4,
}

impl SlotType {
    pub(crate) const fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Reset,
            1 => Self::Normal,
            2 => Self::Final,
            3 => Self::Terminal,
            4 => Self::Truncated,
            _ => panic!("Invalid raw replay buffer slot type"),
        }
    }
}

// One environment's slots, with `^` marking the write head and `t` the `SlotType` row. An
// observation sits one slot ahead of the action and reward it was fed to, which is what lets slot
// `i` read its `next_obs` out of slot `i + 1`.
//
// Episode one ends at slot 3 with `terminated`, so slot 4 keeps its final observation and no
// action; slot 5 starts the next episode. Episode two is truncated the same way at slot 8. The
// head has since reached slot 11, holding `o1` of episode three:
//
// o: [o0, o1, o2, o3, o4, o0, o1, o2, o3, o4, o0, o1]
// a: [a0, a1, a2, a3,   , a0, a1, a2, a3,   , a0,   ]
// r: [r0, r1, r2, r3,   , r0, r1, r2, r3,   , r0,   ]
// t: [N , N , N , Tm, F , N , N , N , Tr, F , N , F ]
// idx:                                            ^
//
// `reset` from there abandons episode three. Slot 11's observation is the `next_obs` of the
// transition at slot 10, so it cannot be overwritten: slot 10 becomes a truncation instead, and
// the new episode starts at slot 12, costing one slot.
//
// o: [o0, o1, o2, o3, o4, o0, o1, o2, o3, o4, o0, o1, o0]
// a: [a0, a1, a2, a3,   , a0, a1, a2, a3,   , a0,   ,   ]
// r: [r0, r1, r2, r3,   , r0, r1, r2, r3,   , r0,   ,   ]
// t: [N , N , N , Tm, F , N , N , N , Tr, F , Tr, F , F ]
// idx:                                                ^
//
// A `terminated` step instead leaves the final observation behind at the slot it claimed and moves
// on, so the reset observation that arrives on the following call gets a slot of its own:
//
// o: [o0, o1, o2,   ]
// a: [a0, a1,   ,   ]
// r: [r0, r1,   ,   ]
// t: [N , Tm, F , R ]
// idx:            ^

/// One environment's slice of the buffer's bookkeeping: its write head, how many of its slots
/// have been written, its row of the `types` table and its window of the priority tree.
///
/// The episode-boundary state machine lives here rather than on [`ReplayBuffer`] so that it can be
/// exercised without a Python interpreter -- see the tests at the bottom of this file -- and so
/// that the two entry points below read as the state machine they are.
struct EnvCursor<'a> {
    head: &'a mut u32,
    len: &'a mut u32,
    types: ArrayViewMut1<'a, u8>,
    /// This environment's slots are `base .. base + capacity` in the tree's flat index space.
    prio_base: usize,
    prios: Option<&'a mut PrioTree>,
    /// The priority a slot is given as it becomes a complete transition.
    max_prio: f32,
}

impl EnvCursor<'_> {
    #[inline]
    fn capacity(&self) -> u32 {
        self.types.len() as u32
    }

    #[inline]
    fn head_type(&self) -> SlotType {
        SlotType::from_raw(self.types[*self.head as usize])
    }

    /// The slot written just before the head, or `None` while the environment is still empty.
    fn prev(&self) -> Option<usize> {
        (0 < *self.len).then(|| ((*self.head + self.capacity() - 1) % self.capacity()) as usize)
    }

    fn set_type(&mut self, slot: usize, ty: SlotType) {
        self.types[slot] = ty as _;
    }

    /// Sets a slot's sampling weight.
    ///
    /// Slots that do not hold a complete transition are given zero, which is what keeps
    /// [`PrioTree::sample`] from ever returning them. Doing so is not merely tidiness: once the
    /// buffer has wrapped, a slot arrives at the head still carrying the priority it was given on
    /// the previous lap, and leaving that in place would make the write head samplable.
    fn set_prio(&mut self, slot: usize, prio: f32) {
        if let Some(prios) = self.prios.as_deref_mut() {
            prios
                .update(self.prio_base + slot, prio)
                .expect("slot is in range, and every priority reaching here is finite and >= 0");
        }
    }

    /// Moves the head one slot on, wrapping and growing the environment's length.
    fn advance(&mut self) {
        *self.head = (*self.head + 1) % self.capacity();
        *self.len = (*self.len + 1).min(self.capacity());
    }

    /// Starts a new episode, returning the slot the reset observation belongs in.
    fn reset(&mut self) -> u32 {
        match self.head_type() {
            // Nothing can reach the head slot, so the observation simply lands there.
            SlotType::Reset => {}
            SlotType::Final => {
                // The head already holds an observation. Where the transition before it is still
                // mid-episode that observation is its `next_obs`, so overwriting it would corrupt
                // the pair; an abandoned episode is exactly a truncation, so close the transition
                // off as one and start a slot later. Otherwise -- a repeated `reset`, or an
                // episode that has already ended -- nothing reads the head and it is free.
                if let Some(prev) = self.prev()
                    && self.types[prev] == SlotType::Normal as u8
                {
                    self.set_type(prev, SlotType::Truncated);
                    self.advance();
                }
            }
            ty => unreachable!("head slot is {ty:?}, which no completed call leaves behind"),
        }

        let head = *self.head;
        self.set_type(head as usize, SlotType::Final);
        self.set_prio(head as usize, 0.0);

        head
    }

    /// Records one transition and advances the head, returning the slot `next_obs` belongs in.
    ///
    /// The caller has already written this step's action and reward to `*self.head`.
    fn step(&mut self, terminated: bool, truncated: bool) -> u32 {
        match self.head_type() {
            // The head holds the observation this transition starts from, so the action and reward
            // just written complete it.
            SlotType::Final => {
                let slot = *self.head as usize;
                self.set_type(
                    slot,
                    match (terminated, truncated) {
                        (true, _) => SlotType::Terminal,
                        (false, true) => SlotType::Truncated,
                        (false, false) => SlotType::Normal,
                    },
                );
                self.set_prio(slot, self.max_prio);
                self.advance();
            }
            // Gymnasium's next-step autoreset: the episode ended on the previous call, so this one
            // carries the new episode's first observation together with a placeholder action and
            // reward. There is no transition to complete, and the placeholders are overwritten by
            // the next call, which lands on this same slot.
            SlotType::Reset => {}
            ty => unreachable!("head slot is {ty:?}, which no completed call leaves behind"),
        }

        let obs_slot = *self.head;
        self.set_type(obs_slot as usize, SlotType::Final);
        self.set_prio(obs_slot as usize, 0.0);

        // Under next-step autoreset the observation just claimed is the episode's *last*, and the
        // reset observation arrives on the following call. Leave the final observation where it is
        // and give the new episode a slot of its own.
        if terminated || truncated {
            self.advance();
            let head = *self.head as usize;
            self.set_type(head, SlotType::Reset);
            self.set_prio(head, 0.0);
        }

        obs_slot
    }
}

/// Rejects `stratified` on a buffer that has no priorities to stratify.
///
/// Stratification slices the *priority* mass, so without priorities there is nothing for it to
/// mean: a uniform draw already spreads evenly over every samplable transition. Refusing it is
/// better than accepting and ignoring it, which would leave a caller believing a variance
/// reduction was in effect that never was.
fn check_stratified(stratified: bool, use_prios: bool) -> PyResult<()> {
    if stratified && !use_prios {
        return Err(PyValueError::new_err(
            "stratified requires use_prios: without priorities there is no mass to spread a batch over",
        ));
    }

    Ok(())
}

/// Checks a per-environment batch against the stored `[env, slot, ..]` array it is written into.
fn check_batch_shape(name: &str, batch: &[usize], array: &[usize]) -> PyResult<()> {
    if batch.len() + 1 != array.len() || batch[0] != array[0] || batch[1..] != array[2..] {
        let mut expected = array.to_vec();
        expected.remove(1);

        return Err(PyValueError::new_err(format!(
            "{name} must have shape {expected:?}, got {batch:?}"
        )));
    }

    Ok(())
}

/// Picks the single frame that [`ReplayBuffer::reset`] and [`ReplayBuffer::save_step`] store out of a
/// batch of observations.
///
/// Without `obs_stack` the batch is one frame per environment, `(num_envs, *obs_shape)`, and is
/// stored as it stands. With `obs_stack` set the caller may hand over either that same single
/// frame -- having already picked the new one out of whatever the environment returned -- or the
/// environment's whole stacked observation, `(num_envs, obs_stack, *obs_shape)`, of which only the
/// newest frame is kept. The newest frame is the last along the stack axis, which is the
/// convention both Gymnasium's `FrameStackObservation` and ALE's own `stack_num` follow.
///
/// Storing only that frame is the point of the whole layout: the earlier frames of the stack are
/// already in the buffer, in the slots before this one, so keeping the stack as handed over would
/// multiply the buffer's memory by `obs_stack` for nothing. [`ReplayBuffer::sample`] reassembles
/// it on the way out.
///
/// The stack axis comes *before* the frame rather than after it so that the frame this picks out
/// is one contiguous run of bytes. Taking it from a trailing stack axis would instead read every
/// byte at a stride of `obs_stack`, turning what should be one memcpy per environment into a
/// scattered per-byte copy touching `obs_stack` times as many cache lines -- on every `save_step`. A
/// model wanting the frames trailing should transpose on the accelerator, where the move costs
/// bandwidth this copy cannot match and is usually folded into the first layer.
fn obs_frame<'a>(
    name: &str,
    batch: DynArrayView<'a>,
    stored: &[usize],
    obs_stack: Option<NonZero<u32>>,
) -> PyResult<DynArrayView<'a>> {
    // `stored` is `[env, slot, ..obs_shape]`; a batch is the same without the slot axis.
    let mut single = stored.to_vec();
    single.remove(1);

    if batch.shape() == single {
        return Ok(batch);
    }

    let Some(stack) = obs_stack.map(|stack| stack.get() as usize) else {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape {single:?}, got {:?}",
            batch.shape()
        )));
    };

    let mut stacked = single.clone();
    stacked.insert(1, stack);

    if batch.shape() == stacked {
        return Ok(batch.index_axis_move(Axis(1), stack - 1));
    }

    Err(PyValueError::new_err(format!(
        "{name} must have shape {single:?} or {stacked:?}, got {:?}",
        batch.shape()
    )))
}

/// What [`ReplayBuffer::sample`] returns: `(indices, prios, obs, act, rewards, terminals,
/// next_obs)`.
type Batch<'py> = (
    Bound<'py, PyArray1<u32>>,  // indices
    Bound<'py, PyArray1<f32>>,  // prios
    Bound<'py, PyAny>,          // obs
    Bound<'py, PyAny>,          // act
    Bound<'py, PyArray1<f32>>,  // rewards
    Bound<'py, PyArray1<bool>>, // terminals
    Bound<'py, PyAny>,          // next_obs
);

/// A circular replay buffer over `num_envs` parallel environments.
#[pyclass]
struct ReplayBuffer {
    rng: Xoshiro256PlusPlus,
    /// Frames per observation that [`ReplayBuffer::sample`] returns, reassembled from consecutive
    /// slots. Also lets [`ReplayBuffer::save_step`] accept an environment's stacked observation and
    /// keep only its newest frame; see [`obs_frame`].
    obs_stack: Option<NonZero<u32>>,
    /// Whether [`ReplayBuffer::sample`] spreads a prioritised batch over the priority mass rather
    /// than drawing each element independently. Only ever true alongside `prios`; see
    /// [`check_stratified`].
    stratified: bool,

    /// Per environment: the slot the next observation is written to.
    env_heads: Box<[u32]>,
    /// Per environment: how many of its slots have been written, saturating at `env_capacity`.
    env_lens: Box<[u32]>,
    /// Where [`ReplayBuffer::drive_envs`] leaves the slot each environment's observation goes
    /// into. Kept on the struct only to keep a per-step allocation out of the write path.
    obs_slots: Box<[u32]>,

    observations: DynArray,
    actions: DynArray,
    rewards: Array2<f32>,
    types: Array2<u8>,

    max_prio: f32,
    max_prio_decay: f32,
    prios: Option<PrioTree>,
}

impl ReplayBuffer {
    /// Runs one environment's boundary state machine per environment, recording in
    /// [`obs_slots`](Self::obs_slots) the slot each environment's incoming observation goes into.
    ///
    /// Both write paths are the same shape -- advance the state machine, then write one
    /// observation per environment into the slot it chose -- and this is the half of that they
    /// share. It is also where the borrow is split: the cursor holds a mutable slice of the
    /// bookkeeping fields for as long as `f` runs, leaving the observation array free for the
    /// caller to write into afterwards.
    fn drive_envs(&mut self, mut f: impl FnMut(usize, &mut EnvCursor<'_>) -> u32) {
        let env_capacity = self.env_capacity();
        let Self {
            env_heads,
            env_lens,
            obs_slots,
            types,
            prios,
            max_prio,
            ..
        } = self;

        for env in 0..env_heads.len() {
            let mut cursor = EnvCursor {
                head: &mut env_heads[env],
                len: &mut env_lens[env],
                types: types.row_mut(env),
                prio_base: env * env_capacity,
                prios: prios.as_mut(),
                max_prio: *max_prio,
            };
            obs_slots[env] = f(env, &mut cursor);
        }
    }
}

#[pymethods]
impl ReplayBuffer {
    /// Allocates a buffer holding `num_envs * env_capacity` transitions.
    ///
    /// `obs_shape` and `act_shape` describe a single observation and action; the environment and
    /// slot axes are prepended internally. Only `float32` and `uint8` are supported for either.
    ///
    /// `obs_stack`, when set, makes [`sample`](Self::sample) return that many consecutive frames
    /// per observation instead of one, and lets [`save_step`](Self::save_step) accept an environment's
    /// stacked observation directly. Only one frame per slot is stored either way, so it does not
    /// change how much memory the buffer needs.
    ///
    /// `stratified` splits the priority mass into `batch_size` equal slices and takes one draw
    /// from each, so a batch spreads over the distribution instead of clumping; the per-draw
    /// marginal is unchanged, only the draws' correlation. It defaults to `use_prios` and requires
    /// it: setting it without priorities raises `ValueError`, since a uniform draw is already
    /// spread over every samplable transition and there is no mass to slice.
    ///
    /// `use_prios` enables prioritised sampling; `max_prio_decay` is covered in
    /// [`update_prios`](Self::update_prios). n-step returns are asked for per draw, not here --
    /// see [`sample`](Self::sample).
    ///
    /// Raises `MemoryError` if the buffer does not fit, `OverflowError` if its flat index would
    /// not fit in a `u32`, and `ValueError` for out-of-range parameters.
    #[new]
    #[pyo3(signature = (
        num_envs, env_capacity,
        *,
        obs_shape, obs_dtype = None,
        act_shape = vec![], act_dtype = None,
        obs_stack = None,
        use_prios = false, max_prio_decay = 0.999, stratified = None,
        seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new<'py>(
        py: Python<'py>,
        num_envs: u32,
        env_capacity: u32,
        mut obs_shape: Vec<usize>,
        obs_dtype: Option<&Bound<'py, PyAny>>,
        mut act_shape: Vec<usize>,
        act_dtype: Option<&Bound<'py, PyAny>>,
        obs_stack: Option<NonZero<u32>>,
        use_prios: bool,
        max_prio_decay: f32,
        stratified: Option<bool>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if num_envs < 1 {
            return Err(PyValueError::new_err("num_envs must be at least one"));
        }

        // Anything below three slots per environment cannot hold a samplable transition at all:
        // the head and the slot its `next_obs` lands in are always spoken for.
        if env_capacity < 3 {
            return Err(PyValueError::new_err("env_capacity must be at least three"));
        }

        if !(0.0 < max_prio_decay && max_prio_decay <= 1.0) {
            return Err(PyValueError::new_err(
                "max_prio_decay must be in the range (0, 1]",
            ));
        }

        let stratified = stratified.unwrap_or(use_prios);
        check_stratified(stratified, use_prios)?;

        let len = num_envs.checked_mul(env_capacity).ok_or_else(|| {
            PyOverflowError::new_err(
                "Cannot represent total replay buffer size in an index sized int",
            )
        })?;

        let rng = match seed {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => rand::make_rng(),
        };

        obs_shape.insert(0, num_envs as _);
        obs_shape.insert(1, env_capacity as _);
        let obs_dtype = obs_dtype.map_or(Ok(DType::F32), DType::from_np)?;

        act_shape.insert(0, num_envs as _);
        act_shape.insert(1, env_capacity as _);
        let act_dtype = act_dtype.map_or(Ok(DType::U8), DType::from_np)?;

        // `len` is `num_envs * env_capacity`, checked above, so neither per-slot table below can
        // reject the buffer it is handed.
        let table_shape = (num_envs as usize, env_capacity as usize);

        py.detach(|| {
            Ok(Self {
                rng,
                obs_stack,
                stratified,

                env_heads: try_zeroed_vec(num_envs as _)?.into_boxed_slice(),
                env_lens: try_zeroed_vec(num_envs as _)?.into_boxed_slice(),
                obs_slots: try_zeroed_vec(num_envs as _)?.into_boxed_slice(),

                observations: DynArray::try_zeros(&obs_shape, obs_dtype)?,
                actions: DynArray::try_zeros(&act_shape, act_dtype)?,
                rewards: Array2::from_shape_vec(table_shape, try_zeroed_vec(len as _)?)
                    .expect("shape is `len` elements"),
                types: Array2::from_shape_vec(table_shape, try_zeroed_vec(len as _)?)
                    .expect("shape is `len` elements"),

                max_prio: 1.0,
                max_prio_decay,
                prios: use_prios.then(|| PrioTree::new(len as _)).transpose()?,
            })
        })
    }

    /// Number of transitions currently stored, summed across environments.
    ///
    /// This counts everything written, which is not the same as the number of transitions that
    /// [`sample`](Self::sample) can actually draw -- the write-head window described in the module
    /// docs is excluded from sampling but included here.
    fn __len__(&self) -> usize {
        self.env_lens.iter().sum::<u32>() as _
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.types.nrows()
    }

    #[getter]
    fn env_capacity(&self) -> usize {
        self.types.ncols()
    }

    #[getter]
    fn total_capacity(&self) -> usize {
        self.rewards.len()
    }

    #[getter]
    fn obs_dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        self.observations.dtype().np(py)
    }

    #[getter]
    fn obs_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.observations.shape()[2..])
    }

    #[getter]
    fn act_dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        self.actions.dtype().np(py)
    }

    #[getter]
    fn act_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.actions.shape()[2..])
    }

    /// Frames per observation that [`sample`](Self::sample) returns, or `None` without a stack.
    #[getter]
    fn obs_stack(&self) -> Option<u32> {
        self.obs_stack.map(NonZero::get)
    }

    #[getter]
    fn use_prios(&self) -> bool {
        self.prios.is_some()
    }

    /// Whether a prioritised batch is spread over the priority mass rather than drawn
    /// independently. Settable: it changes only the draws' correlation, never what is stored, so
    /// there is nothing to keep consistent across a change.
    ///
    /// Setting it on a buffer built without `use_prios` raises `ValueError`.
    #[getter]
    fn stratified(&self) -> bool {
        self.stratified
    }

    #[setter]
    fn set_stratified(&mut self, stratified: bool) -> PyResult<()> {
        check_stratified(stratified, self.prios.is_some())?;
        self.stratified = stratified;

        Ok(())
    }

    /// The priority a newly written transition is given, per
    /// [`update_prios`](Self::update_prios).
    ///
    /// This is the scale [`sample`](Self::sample)'s priorities are drawn against, so an
    /// importance-sampling correction that normalises against the whole buffer rather than the
    /// batch reads it from here.
    #[getter]
    fn max_prio(&self) -> f32 {
        self.max_prio
    }

    /// How fast [`max_prio`](Self::max_prio) decays, per `update_prios` call. Settable, so that a
    /// schedule can track a changing replay ratio -- the half-life is measured in gradient steps.
    #[getter]
    fn max_prio_decay(&self) -> f32 {
        self.max_prio_decay
    }

    #[setter]
    fn set_max_prio_decay(&mut self, max_prio_decay: f32) -> PyResult<()> {
        if !(0.0 < max_prio_decay && max_prio_decay <= 1.0) {
            return Err(PyValueError::new_err(
                "max_prio_decay must be in the range (0, 1]",
            ));
        }
        self.max_prio_decay = max_prio_decay;

        Ok(())
    }

    /// Seeds every environment with its initial observation.
    ///
    /// `obs` is indexed by environment -- shape `(num_envs, *obs_shape)`, or
    /// `(num_envs, obs_stack, *obs_shape)` where the buffer has a frame stack, in which case only
    /// the newest frame is kept. It must already be in the dtype the buffer was built with;
    /// nothing is cast on the way in.
    ///
    /// Normally called once, before the first [`save_step`](Self::save_step): autoreset boundaries during
    /// play are inferred from the `terminated` and `truncated` flags rather than from further
    /// `reset` calls. Calling it again mid-run is still well defined, and means what
    /// `gym.Env.reset` means -- abandon the current episode and start a new one.
    ///
    /// Because successor observations are not stored separately, the observation at an
    /// environment's write head *is* the `next_obs` of the transition before it. Where that
    /// transition exists and is still mid-episode, `reset` does not overwrite the head; it closes
    /// the slot off as a truncation -- which is what an abandoned episode is, cut short with its
    /// final observation known -- and starts the new episode one slot later, costing a single
    /// slot. Where nothing can reach the head observation it is overwritten and no slot is spent:
    /// the opening `reset`, back-to-back `reset` calls, and a `reset` straight after a terminated
    /// or truncated [`save_step`](Self::save_step).
    ///
    /// Raises `TypeError` on a dtype mismatch and `ValueError` on a shape mismatch, in both cases
    /// before anything is written.
    fn reset(&mut self, obs: &Bound<'_, PyAny>) -> PyResult<()> {
        let obs = DynPyArrayLike::extract(self.observations.dtype(), obs)?;
        let obs = obs_frame(
            "obs",
            obs.as_array(),
            self.observations.shape(),
            self.obs_stack,
        )?;

        self.drive_envs(|_, cursor| cursor.reset());
        dyn_write_batch(&mut self.observations, &self.obs_slots, &obs);

        Ok(())
    }

    /// Records one transition per environment and advances each write head.
    ///
    /// Every argument is indexed by environment. `next_obs` takes the same two shapes `obs` does
    /// in [`reset`](Self::reset), and is whatever the environment returned alongside the reward:
    /// under Gymnasium's default `AutoresetMode.NEXT_STEP` that is the episode's *final*
    /// observation wherever `terminated` or `truncated` is set. The call after such a step carries
    /// the new episode's first observation together with an action and a reward the environment
    /// ignored; it completes no transition and does not grow `len(self)`.
    ///
    /// Raises `TypeError` on a dtype mismatch and `ValueError` on a shape mismatch, in both cases
    /// before anything is written.
    fn save_step(
        &mut self,
        actions: &Bound<'_, PyAny>,
        rewards: PyArrayLike1<f32>,
        terminated: PyArrayLike1<bool>,
        truncated: PyArrayLike1<bool>,
        next_obs: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let actions = DynPyArrayLike::extract(self.actions.dtype(), actions)?;
        let actions = actions.as_array();
        check_batch_shape("actions", actions.shape(), self.actions.shape())?;

        let next_obs = DynPyArrayLike::extract(self.observations.dtype(), next_obs)?;
        let next_obs = obs_frame(
            "next_obs",
            next_obs.as_array(),
            self.observations.shape(),
            self.obs_stack,
        )?;

        let num_envs = self.num_envs();
        let (rewards, terminated, truncated) = (
            rewards.as_array(),
            terminated.as_array(),
            truncated.as_array(),
        );
        for (name, len) in [
            ("rewards", rewards.len()),
            ("terminated", terminated.len()),
            ("truncated", truncated.len()),
        ] {
            if len != num_envs {
                return Err(PyValueError::new_err(format!(
                    "{name} must have shape ({num_envs},), got ({len},)"
                )));
            }
        }

        // The action and reward complete the transition the head already holds. On an autoreset
        // call the head holds no such transition and these are placeholders, which the next call
        // overwrites in place.
        dyn_write_batch(&mut self.actions, &self.env_heads, &actions);
        write_batch(&mut self.rewards, &self.env_heads, &rewards);

        self.drive_envs(|env, cursor| cursor.step(terminated[env], truncated[env]));
        dyn_write_batch(&mut self.observations, &self.obs_slots, &next_obs);

        Ok(())
    }

    /// Draws `batch_size` transitions, returning
    /// `(indices, prios, obs, act, rewards, terminals, next_obs)`.
    ///
    /// `indices` are the flat indices described in the module docs, to be handed back to
    /// [`update_prios`](Self::update_prios). `prios` holds each drawn transition's stored priority
    /// as it stands, and `1.0` throughout for a buffer sampling uniformly. It is raw rather than
    /// normalised so that the caller can pick its own denominator: [`sum_prios`](Self::sum_prios)
    /// turns it into the sampling probability, and [`max_prio`](Self::max_prio) into a weight
    /// scaled against the whole buffer rather than against the batch that came back.
    ///
    /// `rewards` is the accumulated `n_steps` return and `next_obs` the observation `n_steps`
    /// later, with `terminals` set only for true episode ends: a transition truncated by a time
    /// limit should still be bootstrapped from, so it is not marked terminal here.
    #[pyo3(signature = (
        batch_size,
        *,
        n_steps = NonZero::new(1).unwrap(),
        discount = None,
    ))]
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: usize,
        n_steps: NonZero<u32>,
        discount: Option<f32>,
    ) -> PyResult<Batch<'py>> {
        if n_steps.get() != 1 && discount.is_none() {
            return Err(PyValueError::new_err(
                "discount must be set if n_steps is not one",
            ));
        }
        let discount = discount.unwrap_or(1.0);

        if !(0.0 < discount && discount <= 1.0) {
            return Err(PyValueError::new_err(
                "discount must be in the range (0, 1]",
            ));
        }

        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be at least one"));
        }

        let (num_envs, capacity) = (self.num_envs(), self.env_capacity());
        let spec = sampling::DrawSpec {
            obs_stack: self.obs_stack.map_or(1, NonZero::get),
            n_steps: n_steps.get(),
            discount,
            stratified: self.stratified,
        };

        // Split the borrow: `envs` reads `types` and `rewards` for as long as the draw runs, while
        // the draw also needs `&mut rng`.
        let Self {
            rng,
            env_heads,
            env_lens,
            types,
            rewards,
            prios,
            ..
        } = self;

        let envs: Vec<_> = (0..num_envs)
            .map(|env| {
                sampling::EnvSlots::new(
                    env_heads[env],
                    env_lens[env],
                    types.row(env),
                    rewards.row(env),
                    spec,
                )
            })
            .collect();

        let batch = sampling::draw_batch(rng, &envs, prios.as_ref(), batch_size, spec)?;

        // `[batch, obs_stack, ..obs_shape]`, with the stack axis only where the buffer has one.
        let mut obs_shape = vec![batch_size];
        obs_shape.extend(self.obs_stack.map(|stack| stack.get() as usize));
        obs_shape.extend_from_slice(&self.observations.shape()[2..]);

        let mut act_shape = vec![batch_size];
        act_shape.extend_from_slice(&self.actions.shape()[2..]);

        let act_slots: Vec<u32> = batch.draws.iter().map(|draw| draw.index).collect();
        let obs = &self.observations;

        Ok((
            PyArray1::from_iter(py, batch.draws.iter().map(|draw| draw.index)),
            PyArray1::from_iter(py, batch.draws.iter().map(|draw| draw.prio)),
            dyn_gather(py, obs, &obs_shape, &batch.obs_slots, capacity),
            dyn_gather(py, &self.actions, &act_shape, &act_slots, capacity),
            PyArray1::from_iter(py, batch.draws.iter().map(|draw| draw.ret)),
            PyArray1::from_iter(py, batch.draws.iter().map(|draw| draw.terminal)),
            dyn_gather(py, obs, &obs_shape, &batch.next_obs_slots, capacity),
        ))
    }

    /// Replaces the priorities at `indices` with `prios`.
    ///
    /// Both arrays must be the same length, and the priorities must be finite and non-negative.
    /// Duplicate indices are applied in order, so the last value for an index wins. No exponent is
    /// applied here -- see the module docs on where `alpha` belongs.
    ///
    /// This also refreshes the priority that newly written transitions are given, as the larger of
    /// the highest priority in `prios` and the previous value decayed by `max_prio_decay`. Decaying
    /// rather than keeping a running maximum matters over a long run: TD errors shrink as the agent
    /// improves, so a maximum that never falls would keep handing new transitions a priority drawn
    /// from early training and heavily oversample them. Note that the decay is applied per call, so
    /// its half-life is measured in gradient steps and shifts with the replay ratio.
    ///
    /// Raises `ValueError` if this buffer was built without priorities.
    fn update_prios(
        &mut self,
        indices: PyArrayLike1<u32>,
        prios: PyArrayLike1<f32>,
    ) -> PyResult<()> {
        let Some(rb_prios) = &mut self.prios else {
            return Err(PyValueError::new_err(
                "Priorities are not used in this replay buffer",
            ));
        };
        let (indices, prios) = (indices.as_array(), prios.as_array());
        if indices.len() != prios.len() {
            return Err(PyValueError::new_err(
                "indices and priorities array lengths are different",
            ));
        }

        // Validate the whole batch before touching the tree. Failing part way through would leave
        // the buffer sampling from a mix of old and new priorities, with nothing to tell the
        // caller how far the update got.
        let len = rb_prios.len();
        for (&i, &p) in indices.iter().zip(prios) {
            if len <= i as usize {
                return Err(PyIndexError::new_err(format!(
                    "index {i} is out of range for a buffer of {len} transitions"
                )));
            }
            if !p.is_finite() || p < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "priority {p} is not finite and non-negative"
                )));
            }
        }

        let mut max_prio = self.max_prio * self.max_prio_decay;
        for (&i, &p) in indices.iter().zip(prios) {
            max_prio = max_prio.max(p);
            rb_prios
                .update(i as _, p)
                .expect("the whole batch was validated above");
        }
        self.max_prio = max_prio;

        Ok(())
    }

    /// Sum of every stored priority: the normalising constant that turns [`sample`](Self::sample)'s
    /// priorities into probabilities.
    ///
    /// Raises `ValueError` if this buffer was built without priorities.
    #[getter]
    fn sum_prios(&self) -> PyResult<f32> {
        let Some(prios) = &self.prios else {
            return Err(PyValueError::new_err(
                "Priorities are not used in this replay buffer",
            ));
        };

        Ok(prios.total())
    }
}

#[pymodule]
mod expreplay {
    #[pymodule_export]
    use super::ReplayBuffer;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_batch_writes_one_slot_per_environment() {
        // Two environments, two slots each, two-element observations.
        let mut dst = ArrayD::<u8>::zeros(IxDyn(&[2, 2, 2]));
        let src = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1, 2, 3, 4]).unwrap();

        write_batch(&mut dst, &[1, 0], &src);

        assert_eq!(dst.into_raw_vec_and_offset().0, [0, 0, 1, 2, 3, 4, 0, 0]);
    }

    #[test]
    fn test_write_batch_agrees_across_the_parallel_threshold() {
        // The rayon branch only kicks in for observation-sized batches, so nothing else here
        // reaches it.
        let cols = PAR_COPY_MIN_ELEMS;
        let mut dst = ArrayD::<u8>::zeros(IxDyn(&[2, 2, cols]));
        let src = ArrayD::from_shape_fn(IxDyn(&[2, cols]), |i| (i[0] + i[1]) as _);

        write_batch(&mut dst, &[1, 0], &src);

        assert_eq!(
            dst.index_axis(Axis(0), 0).index_axis(Axis(0), 1),
            src.index_axis(Axis(0), 0)
        );
        assert_eq!(
            dst.index_axis(Axis(0), 1).index_axis(Axis(0), 0),
            src.index_axis(Axis(0), 1)
        );
        assert!(
            dst.index_axis(Axis(0), 0)
                .index_axis(Axis(0), 0)
                .iter()
                .all(|&v| v == 0)
        );
    }

    #[test]
    fn test_check_batch_shape() {
        // `[env, slot, 4, 4]` is fed by an `[env, 4, 4]` batch.
        let stored = [2, 8, 4, 4];
        check_batch_shape("obs", &[2, 4, 4], &stored).unwrap();
        check_batch_shape("obs", &[3, 4, 4], &stored).unwrap_err(); // wrong env count
        check_batch_shape("obs", &[2, 4, 5], &stored).unwrap_err(); // wrong tail
        check_batch_shape("obs", &[2, 8, 4, 4], &stored).unwrap_err(); // slot axis included
        check_batch_shape("obs", &[], &stored).unwrap_err(); // must not index into an empty shape

        // A scalar action is stored as `[env, slot]` and fed by an `[env]` batch.
        check_batch_shape("act", &[2], &[2, 8]).unwrap();
        check_batch_shape("act", &[2, 1], &[2, 8]).unwrap_err();
    }

    /// [`obs_frame`], with the `PyErr` kept out of the assertions -- formatting one needs an
    /// interpreter, which these tests do not have.
    fn frame(batch: &ArrayD<u8>, stored: &[usize], stack: Option<u32>) -> Option<ArrayD<u8>> {
        let batch = DynArrayView::U8(batch.view());
        let stack = stack.map(|stack| NonZero::new(stack).unwrap());

        match obs_frame("obs", batch, stored, stack) {
            Ok(DynArrayView::U8(frame)) => Some(frame.to_owned()),
            Ok(DynArrayView::F32(_)) => unreachable!("the batch went in as u8"),
            Err(_) => None,
        }
    }

    #[test]
    fn test_obs_frame_without_a_stack_takes_the_batch_as_it_stands() {
        // Stored as `[env, slot, 2]`, so a batch is `[env, 2]`.
        let stored = [2, 8, 2];
        let batch = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1u8, 2, 3, 4]).unwrap();

        assert_eq!(frame(&batch, &stored, None).unwrap(), batch);

        // A stacked batch is not accepted when the buffer was not built for one.
        let stacked = ArrayD::<u8>::zeros(IxDyn(&[2, 3, 2]));
        assert!(frame(&stacked, &stored, None).is_none());
    }

    #[test]
    fn test_obs_frame_keeps_only_the_newest_frame_of_a_stack() {
        let stored = [2, 8, 2];

        // `[env, stack, 2]`, with the frame index in the ones digit: the newest frame is the last
        // along the stack axis, so `[[12, 22], [32, 42]]` is what gets stored.
        let stacked = ArrayD::from_shape_fn(IxDyn(&[2, 3, 2]), |i| {
            (10 * (2 * i[0] + i[2] + 1) + i[1]) as _
        });
        let expected = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![12u8, 22, 32, 42]).unwrap();
        assert_eq!(frame(&stacked, &stored, Some(3)).unwrap(), expected);

        // A caller that has already picked the newest frame out itself is equally welcome.
        let single = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1u8, 2, 3, 4]).unwrap();
        assert_eq!(frame(&single, &stored, Some(3)).unwrap(), single);

        // Anything else is rejected: a stack of the wrong depth, a *trailing* stack axis, and a
        // batch with the wrong number of environments.
        assert!(frame(&ArrayD::zeros(IxDyn(&[2, 2, 2])), &stored, Some(3)).is_none());
        assert!(frame(&ArrayD::zeros(IxDyn(&[2, 2, 3])), &stored, Some(3)).is_none());
        assert!(frame(&ArrayD::zeros(IxDyn(&[3, 3, 2])), &stored, Some(3)).is_none());
    }

    /// One environment's worth of state, driven through [`EnvCursor`] the way `ReplayBuffer` does.
    struct TestEnv {
        head: u32,
        len: u32,
        types: Array1<u8>,
        prios: PrioTree,
    }

    impl TestEnv {
        fn new(capacity: usize) -> Self {
            Self {
                head: 0,
                len: 0,
                types: Array1::zeros(capacity),
                prios: PrioTree::new(capacity).unwrap(),
            }
        }

        fn cursor(&mut self) -> EnvCursor<'_> {
            EnvCursor {
                head: &mut self.head,
                len: &mut self.len,
                types: self.types.view_mut(),
                prio_base: 0,
                prios: Some(&mut self.prios),
                max_prio: 1.0,
            }
        }

        fn reset(&mut self) {
            self.cursor().reset();
        }

        fn step(&mut self, terminated: bool, truncated: bool) -> u32 {
            self.cursor().step(terminated, truncated)
        }

        fn types(&self) -> Vec<SlotType> {
            self.types.iter().copied().map(SlotType::from_raw).collect()
        }
    }

    use SlotType::{Final, Normal, Reset, Terminal, Truncated};

    #[test]
    fn test_reset_then_steps_fill_consecutive_slots() {
        let mut env = TestEnv::new(6);
        env.reset();
        assert_eq!((env.head, env.len), (0, 0));

        for step in 0..3 {
            // `next_obs` lands one slot on from the transition the step just completed.
            assert_eq!(env.step(false, false), step + 1);
        }

        assert_eq!(env.types(), [Normal, Normal, Normal, Final, Reset, Reset]);
        assert_eq!((env.head, env.len), (3, 3));
    }

    #[test]
    fn test_repeated_reset_spends_no_slot() {
        // Nothing can read the head observation yet, so a second and third `reset` just overwrite
        // it. This used to fall through the state machine and panic.
        let mut env = TestEnv::new(4);
        for _ in 0..3 {
            env.reset();
            assert_eq!((env.head, env.len), (0, 0));
            assert_eq!(env.types(), [Final, Reset, Reset, Reset]);
        }
    }

    #[test]
    fn test_reset_mid_episode_truncates_and_spends_one_slot() {
        let mut env = TestEnv::new(6);
        env.reset();
        env.step(false, false);
        env.step(false, false);
        assert_eq!(env.types(), [Normal, Normal, Final, Reset, Reset, Reset]);

        // Slot 2 holds slot 1's `next_obs`, so it cannot be reused: slot 1 becomes a truncation
        // and the new episode starts at slot 3.
        env.reset();
        assert_eq!(env.types(), [Normal, Truncated, Final, Final, Reset, Reset]);
        assert_eq!((env.head, env.len), (3, 3));
    }

    #[test]
    fn test_reset_after_an_episode_end_spends_no_slot() {
        let mut env = TestEnv::new(6);
        env.reset();
        env.step(true, false);
        assert_eq!(env.types(), [Terminal, Final, Reset, Reset, Reset, Reset]);
        let (head, len) = (env.head, env.len);

        // The head is the empty slot the termination left behind, so the reset observation lands
        // straight in it.
        env.reset();
        assert_eq!(env.types(), [Terminal, Final, Final, Reset, Reset, Reset]);
        assert_eq!((env.head, env.len), (head, len));
    }

    #[test]
    fn test_terminated_step_keeps_the_final_observation() {
        let mut env = TestEnv::new(6);
        env.reset();
        env.step(false, false);

        // Slot 2 keeps the episode's final observation -- it is slot 1's `next_obs` -- and slot 3
        // is left empty for the reset observation the next call brings.
        assert_eq!(env.step(true, false), 2);
        assert_eq!(env.types(), [Normal, Terminal, Final, Reset, Reset, Reset]);
        assert_eq!((env.head, env.len), (3, 3));

        // That next call carries a placeholder action and reward and completes no transition.
        assert_eq!(env.step(false, false), 3);
        assert_eq!(env.types(), [Normal, Terminal, Final, Final, Reset, Reset]);
        assert_eq!((env.head, env.len), (3, 3));

        assert_eq!(env.step(false, false), 4);
        assert_eq!(env.types(), [Normal, Terminal, Final, Normal, Final, Reset]);
    }

    #[test]
    fn test_truncated_step_matches_terminated_but_for_the_slot_type() {
        let mut env = TestEnv::new(6);
        env.reset();
        assert_eq!(env.step(false, true), 1);
        assert_eq!(env.types(), [Truncated, Final, Reset, Reset, Reset, Reset]);
        assert_eq!((env.head, env.len), (2, 2));
    }

    #[test]
    fn test_only_complete_transitions_are_samplable() {
        // Every slot that does not hold a whole transition must carry a zero priority, or the
        // sum-tree would hand out an index into an unwritten or half-written slot.
        let mut env = TestEnv::new(6);
        env.reset();
        for _ in 0..3 {
            env.step(false, false);
        }
        env.step(true, false);

        let samplable: Vec<bool> = env.prios.prios().iter().map(|&p| p > 0.0).collect();
        let complete: Vec<bool> = env
            .types()
            .iter()
            .map(|ty| matches!(ty, Normal | Terminal | Truncated))
            .collect();
        assert_eq!(samplable, complete);
        assert_eq!(samplable, [true, true, true, true, false, false]);
    }

    #[test]
    fn test_wrapping_clears_the_priority_of_recycled_slots() {
        // Once the buffer laps, a slot reaches the head still carrying the priority it was given
        // on the previous lap. It has to be cleared as the head claims it.
        let mut env = TestEnv::new(4);
        env.reset();
        for _ in 0..10 {
            env.step(false, false);
            let head = env.head as usize;
            assert_eq!(
                env.prios.prios()[head],
                0.0,
                "head slot {head} is samplable"
            );
            assert!(env.types()[head] == Final);
        }
        assert_eq!(env.len, 4);

        // Three complete transitions and the head, all lap after lap.
        let samplable = env.prios.prios().iter().filter(|&&p| p > 0.0).count();
        assert_eq!(samplable, 3);
    }
}
