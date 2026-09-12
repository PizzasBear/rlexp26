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
//! `save_step` picks its frame out with a memcpy per environment instead of a stride-`obs_stack`
//! byte scatter, and `sample` builds a stack as `obs_stack` back-to-back memcpys. A model wanting
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

use std::{borrow::Cow, num::NonZero, ops::RangeInclusive};

use ndarray::prelude::*;
use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
use thiserror::Error;

mod dyn_array;
mod prio_tree;
mod pyffi;
mod sampling;
mod utils;

use crate::dyn_array::{DType, DynArray, DynArrayView, dyn_gather, dyn_write_batch};
use crate::prio_tree::PrioTree;
use crate::utils::{AllocationError, try_zeroed_vec, write_batch};

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

/// One drawn batch: what [`ReplayBuffer::sample`] returns.
///
/// Every field is indexed by draw. `obs` and `next_obs` carry a leading stack axis where the
/// buffer has a frame stack, so they are `[batch, obs_stack, ..obs_shape]` against `actions`'
/// `[batch, ..act_shape]`.
pub struct Batch {
    pub indices: Array1<u32>,
    pub prios: Array1<f32>,
    pub obs: DynArray,
    pub actions: DynArray,
    pub rewards: Array1<f32>,
    pub terminals: Array1<bool>,
    pub next_obs: DynArray,
}

/// Everything [`ReplayBuffer`] rejects a call for.
///
/// The variants are split by what kind of mistake they represent rather than by which method
/// raised them, so that the Python boundary can map each onto the exception a caller expects
/// without knowing where it came from; see the `From<ReplayBufferError>` the `pyffi` module adds.
#[derive(Error, Debug)]
pub enum ReplayBufferError {
    #[error("{0}")]
    AllocationError(#[from] AllocationError),
    #[error("{0}")]
    InvalidArgument(Cow<'static, str>),
    #[error("{0}")]
    InvalidShape(Cow<'static, str>),
    #[error("{0}")]
    ArgumentOverflow(Cow<'static, str>),
    #[error("{0}")]
    RequiresPriorities(Cow<'static, str>),
    #[error("{0}")]
    IndexOutOfRange(Cow<'static, str>),
    /// The draw itself failed; `sampling::DrawError` says whether that is the caller's doing.
    #[error("{0}")]
    Draw(#[from] sampling::DrawError),
}

impl ReplayBufferError {
    fn invalid_argument(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::InvalidArgument(msg.into())
    }
    fn invalid_shape(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::InvalidShape(msg.into())
    }
    fn argument_overflow(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::ArgumentOverflow(msg.into())
    }
    fn requires_priorities(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::RequiresPriorities(msg.into())
    }
    fn index_out_of_range(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::IndexOutOfRange(msg.into())
    }
}

pub type ReplayBufferResult<T> = Result<T, ReplayBufferError>;

/// How a [`ReplayBuffer`] is to be built.
///
/// The fields are private and the `with_*` methods are the only way to set them, so that there is
/// one construction path to validate rather than a struct literal that can also be written by
/// hand. [`ReplayBuffer::new`] takes it by reference and reads it directly.
#[derive(Debug, Clone, Copy)]
pub struct ReplayBufferSpec<'a> {
    obs_shape: &'a [usize],
    obs_dtype: DType,
    act_shape: &'a [usize],
    act_dtype: DType,
    obs_stack: Option<NonZero<u32>>,
    use_prios: bool,
    max_prio: f32,
    max_prio_decay: f32,
    stratified: Option<bool>,
    seed: Option<u64>,
}

impl<'a> ReplayBufferSpec<'a> {
    pub fn new(
        obs_shape: &'a [usize],
        obs_dtype: DType,
        act_shape: &'a [usize],
        act_dtype: DType,
    ) -> Self {
        Self {
            obs_shape,
            obs_dtype,
            act_shape,
            act_dtype,
            obs_stack: None,
            use_prios: false,
            max_prio: 1.0,
            max_prio_decay: 0.999,
            stratified: None,
            seed: None,
        }
    }

    pub fn with_obs_stack(mut self, obs_stack: Option<NonZero<u32>>) -> Self {
        self.obs_stack = obs_stack;
        self
    }

    pub fn with_use_prios(mut self, use_prios: bool) -> Self {
        self.use_prios = use_prios;
        self
    }

    pub fn with_max_prio(mut self, max_prio: f32) -> Self {
        self.max_prio = max_prio;
        self
    }

    pub fn with_max_prio_decay(mut self, max_prio_decay: f32) -> Self {
        self.max_prio_decay = max_prio_decay;
        self
    }

    pub fn with_stratified(mut self, stratified: Option<bool>) -> Self {
        self.stratified = stratified;
        self
    }

    pub fn with_seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }
}

/// What one call to [`ReplayBuffer::sample`] asks for. Private fields for the reason
/// [`ReplayBufferSpec`]'s are private.
#[derive(Clone, Copy, Debug)]
pub struct ReplayBufferSampleParams {
    n_steps: NonZero<u32>,
    discount: Option<f32>,
}

impl ReplayBufferSampleParams {
    pub const fn new() -> Self {
        Self {
            n_steps: NonZero::<u32>::MIN,
            discount: None,
        }
    }

    pub const fn with_n_steps(mut self, n_steps: NonZero<u32>) -> Self {
        self.n_steps = n_steps;
        self
    }

    pub const fn with_discount(mut self, discount: Option<f32>) -> Self {
        self.discount = discount;
        self
    }

    /// The factor a rollout actually discounts by.
    ///
    /// An unset `discount` means one step per draw and so nothing to discount;
    /// [`ReplayBuffer::sample`] rejects a multi-step draw that left it unset rather than quietly
    /// accumulating an undiscounted return.
    pub const fn discount_or_one(&self) -> f32 {
        match self.discount {
            Some(discount) => discount,
            None => 1.0,
        }
    }
}

impl Default for ReplayBufferSampleParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Rejects a `max_prio_decay` outside `(0, 1]`. Shared by the constructor and the setter, which
/// have to agree: a value the setter refuses must not be reachable by building a buffer with it.
fn check_max_prio_decay(max_prio_decay: f32) -> ReplayBufferResult<()> {
    if !(0.0 < max_prio_decay && max_prio_decay <= 1.0) {
        return Err(ReplayBufferError::invalid_argument(
            "max_prio_decay must be in the range (0, 1]",
        ));
    }

    Ok(())
}

/// Rejects `stratified` on a buffer with no priorities. Shared by the constructor and the setter
/// for the same reason [`check_max_prio_decay`] is.
fn check_stratified(stratified: bool, use_prios: bool) -> ReplayBufferResult<()> {
    if stratified && !use_prios {
        return Err(ReplayBufferError::requires_priorities(
            "stratified requires use_prios: without priorities there is no mass to spread a batch over",
        ));
    }

    Ok(())
}

/// Checks a per-environment batch against the stored `[env, slot, ..]` array it is written into.
fn check_batch_shape(name: &str, batch: &[usize], array: &[usize]) -> ReplayBufferResult<()> {
    if batch.len() + 1 != array.len() || batch[0] != array[0] || batch[1..] != array[2..] {
        let mut expected = array.to_vec();
        expected.remove(1);

        return Err(ReplayBufferError::invalid_shape(format!(
            "{name} must have shape {expected:?}, got {batch:?}"
        )));
    }

    Ok(())
}

/// Picks the single frame that [`ReplayBuffer::reset`] and [`ReplayBuffer::save_step`] store out of
/// a batch of observations.
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
/// The stack axis comes *before* the frame rather than after it so that the frame this picks out is
/// one contiguous run of bytes. Taking it from a trailing stack axis would instead read every byte
/// at a stride of `obs_stack`, turning what should be one memcpy per environment into a scattered
/// per-byte copy touching `obs_stack` times as many cache lines -- on every `save_step`. A model
/// wanting the frames trailing should transpose on the accelerator, where the move costs bandwidth
/// this copy cannot match and is usually folded into the first layer.
fn obs_frame<'a>(
    name: &'static str,
    batch: DynArrayView<'a>,
    stored: &[usize],
    obs_stack: Option<NonZero<u32>>,
) -> ReplayBufferResult<DynArrayView<'a>> {
    // `stored` is `[env, slot, ..obs_shape]`; a batch is the same without the slot axis.
    let mut single = stored.to_vec();
    single.remove(1);

    if batch.shape() == single {
        return Ok(batch);
    }

    let Some(stack) = obs_stack.map(|stack| stack.get() as usize) else {
        return Err(ReplayBufferError::invalid_shape(format!(
            "{name} must have shape {single:?}, got {:?}",
            batch.shape()
        )));
    };

    let mut stacked = single.clone();
    stacked.insert(1, stack);

    if batch.shape() == stacked {
        return Ok(batch.index_axis_move(Axis(1), stack - 1));
    }

    Err(ReplayBufferError::invalid_shape(format!(
        "{name} must have shape {single:?} or {stacked:?}, got {:?}",
        batch.shape()
    )))
}

/// The ages a draw may land on in an environment that has written `len` slots, or `None` when it
/// cannot serve one yet.
///
/// Ages count back from the write head: age 0 is the head, age 1 the transition completed most
/// recently. Complete transitions sit at ages `1 ..= len` and the oldest observation at age `len`
/// -- one further back than the oldest transition, since the head's observation is the previous
/// transition's `next_obs` -- so `obs_len` below is `len + 1`, clamped to `capacity`.
///
/// That is a bound on the oldest age, not a count of observations, and the two part company for
/// one call after an episode ends: the head is then an empty `Reset` slot, so there are `len`
/// observations, at ages `1 ..= len`, with age 0 holding nothing. The bound is unaffected, since
/// the oldest age is `len` either way, and nothing ever reads the empty age 0. A rollout could
/// only end there starting from age `n_steps`, which means walking through age 1 -- and with the
/// head on a `Reset` slot, age 1 is the observation the ended episode was parked on and age 2 the
/// transition that ended it, which [`ReplayBuffer::rollout`] stops at before it gets there.
///
/// A draw at age `a` needs two things. Its frame stack reads backwards, ages
/// `a ..= a + obs_stack - 1`, which must stay inside the written observations rather than wrapping
/// past the oldest one onto the newest. Its rollout reads forwards, ages `a ..= a - n_steps + 1`,
/// which must stay on complete transitions, so `a >= n_steps`. The `next_obs` the rollout ends on
/// may be the head itself, at age 0, which is exactly what that observation is there for.
fn valid_ages(
    len: u32,
    capacity: u32,
    obs_stack: u32,
    n_steps: u32,
) -> Option<RangeInclusive<u32>> {
    let obs_len = (len + 1).min(capacity);
    let oldest = obs_len.checked_sub(obs_stack)?;

    (n_steps <= oldest).then_some(n_steps..=oldest)
}

/// A circular replay buffer over `num_envs` parallel environments.
pub struct ReplayBuffer {
    rng: Xoshiro256PlusPlus,
    /// Frames per observation that [`ReplayBuffer::sample`] returns, reassembled from consecutive
    /// slots. Also lets [`ReplayBuffer::save_step`] accept an environment's stacked observation and
    /// keep only its newest frame; see [`obs_frame`].
    obs_stack: Option<NonZero<u32>>,
    /// Whether [`ReplayBuffer::sample`] spreads a prioritised batch over the priority mass rather
    /// than drawing each element independently. Only ever true alongside `prios`
    stratified: bool,

    /// Per environment: the slot the next observation is written to.
    env_heads: Box<[u32]>,
    /// Per environment: how many of its slots have been written, saturating at `env_capacity`.
    env_lens: Box<[u32]>,
    /// Scratch: the slot each environment's incoming observation goes into, as the boundary state
    /// machine chooses it. Both write paths are the same shape -- run the state machine once per
    /// environment, then write one observation per environment into the slot it picked -- and this
    /// carries the first half of that to the second. On the struct only to keep a per-step
    /// allocation out of the write path.
    obs_slots: Box<[u32]>,

    observations: DynArray,
    actions: DynArray,
    rewards: Array2<f32>,
    types: Array2<u8>,

    max_prio: f32,
    max_prio_decay: f32,
    prios: Option<PrioTree>,
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

impl ReplayBuffer {
    // Construction.

    pub fn new(
        num_envs: NonZero<u32>,
        env_capacity: u32,
        spec: &'_ ReplayBufferSpec<'_>,
    ) -> ReplayBufferResult<Self> {
        if env_capacity < 3 {
            return Err(ReplayBufferError::invalid_argument(
                "env_capacity must be at least three",
            ));
        }

        // The starting `max_prio` is held to the rule a priority is held to in `update_prios`,
        // since it is the same quantity, seeded rather than computed. Zero is therefore allowed,
        // and means new transitions stay unsamplable until something writes a priority. Unlike
        // `max_prio_decay` there is no setter to keep in agreement, so this is checked inline.
        if !spec.max_prio.is_finite() || spec.max_prio < 0.0 {
            return Err(ReplayBufferError::invalid_argument(format!(
                "max_prio {} must be finite and non-negative",
                spec.max_prio
            )));
        }

        check_max_prio_decay(spec.max_prio_decay)?;

        // Stratification slices the *priority* mass, so without priorities there is nothing for
        // it to mean: a uniform draw already spreads evenly over every samplable transition.
        // Refusing it beats accepting and ignoring it, which would leave a caller believing a
        // variance reduction was in effect that never was.
        let stratified = spec.stratified.unwrap_or(spec.use_prios);
        check_stratified(stratified, spec.use_prios)?;

        let len = num_envs.get().checked_mul(env_capacity).ok_or_else(|| {
            ReplayBufferError::argument_overflow(
                "num_envs * env_capacity does not fit in the u32 index sample returns",
            )
        })?;

        let rng = match spec.seed {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => rand::make_rng(),
        };

        let mut obs_shape = Vec::with_capacity(spec.obs_shape.len() + 2);
        obs_shape.push(num_envs.get() as _);
        obs_shape.push(env_capacity as _);
        obs_shape.extend_from_slice(spec.obs_shape);

        let mut act_shape = Vec::with_capacity(spec.act_shape.len() + 2);
        act_shape.push(num_envs.get() as _);
        act_shape.push(env_capacity as _);
        act_shape.extend_from_slice(spec.act_shape);

        // `len` is `num_envs * env_capacity`, checked above, so neither per-slot table below can
        // reject the buffer it is handed.
        let table_shape = (num_envs.get() as usize, env_capacity as usize);

        Ok(Self {
            rng,
            obs_stack: spec.obs_stack,
            stratified,

            env_heads: try_zeroed_vec(num_envs.get() as _)?.into_boxed_slice(),
            env_lens: try_zeroed_vec(num_envs.get() as _)?.into_boxed_slice(),
            obs_slots: try_zeroed_vec(num_envs.get() as _)?.into_boxed_slice(),

            observations: DynArray::try_zeros(&obs_shape, spec.obs_dtype)?,
            actions: DynArray::try_zeros(&act_shape, spec.act_dtype)?,
            rewards: Array2::from_shape_vec(table_shape, try_zeroed_vec(len as _)?)
                .expect("shape is `len` elements"),
            types: Array2::from_shape_vec(table_shape, try_zeroed_vec(len as _)?)
                .expect("shape is `len` elements"),

            max_prio: spec.max_prio,
            max_prio_decay: spec.max_prio_decay,
            prios: if spec.use_prios {
                Some(PrioTree::new(len as _)?)
            } else {
                None
            },
        })
    }

    // What the buffer holds. `len` counts everything written, including the window around
    // each write head that `sample` will not draw from.

    /// Number of transitions currently stored, summed across environments.
    ///
    /// This counts everything written, which is not the same as the number of transitions that
    /// [`sample`](Self::sample) can actually draw -- the write-head window described in the module
    /// docs is excluded from sampling but included here.
    pub fn len(&self) -> usize {
        self.env_lens.iter().sum::<u32>() as _
    }

    /// Whether nothing has been written yet. A buffer is empty until its first
    /// [`reset`](Self::reset) or [`save_step`](Self::save_step), and never again.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_envs(&self) -> usize {
        self.types.nrows()
    }

    pub fn env_capacity(&self) -> usize {
        self.types.ncols()
    }

    pub fn total_capacity(&self) -> usize {
        self.rewards.len()
    }

    pub fn obs_dtype(&self) -> DType {
        self.observations.dtype()
    }

    pub fn obs_shape(&self) -> &[usize] {
        &self.observations.shape()[2..]
    }

    pub fn act_dtype(&self) -> DType {
        self.actions.dtype()
    }

    pub fn act_shape(&self) -> &[usize] {
        &self.actions.shape()[2..]
    }

    pub fn obs_stack(&self) -> Option<u32> {
        self.obs_stack.map(NonZero::get)
    }

    pub fn use_prios(&self) -> bool {
        self.prios.is_some()
    }

    /// Whether a prioritised batch is spread over the priority mass rather than drawn
    /// independently. Settable: it changes only the draws' correlation, never what is stored, so
    /// there is nothing to keep consistent across a change.
    ///
    /// Setting it on a buffer built without priorities is a
    /// [`RequiresPriorities`](ReplayBufferError::RequiresPriorities).
    pub fn stratified(&self) -> bool {
        self.stratified
    }

    pub fn set_stratified(&mut self, stratified: bool) -> ReplayBufferResult<()> {
        check_stratified(stratified, self.use_prios())?;
        self.stratified = stratified;

        Ok(())
    }

    /// The priority a newly written transition is given, per
    /// [`update_prios`](Self::update_prios).
    ///
    /// This is the scale [`sample`](Self::sample)'s priorities are drawn against, so an
    /// importance-sampling correction that normalises against the whole buffer rather than the
    /// batch reads it from here.
    ///
    /// Its starting value is a [`ReplayBufferSpec`] field -- a run resuming from a checkpoint
    /// carries it across, so that the transitions its refilled buffer writes are not all handed
    /// the priority of a fresh one -- but there is no setter, unlike
    /// [`max_prio_decay`](Self::max_prio_decay). Every priority already written was written
    /// against some value of this one, and moving it mid-run silently reweights new transitions
    /// against transitions that were new under the old scale.
    pub fn max_prio(&self) -> f32 {
        self.max_prio
    }

    /// How fast [`max_prio`](Self::max_prio) decays, per `update_prios` call. Settable, so that a
    /// schedule can track a changing replay ratio -- the half-life is measured in gradient steps.
    pub fn max_prio_decay(&self) -> f32 {
        self.max_prio_decay
    }

    pub fn set_max_prio_decay(&mut self, max_prio_decay: f32) -> ReplayBufferResult<()> {
        check_max_prio_decay(max_prio_decay)?;
        self.max_prio_decay = max_prio_decay;

        Ok(())
    }

    // Slot bookkeeping: the primitives the write and read paths below both move. Nothing outside
    // this file ever writes `types` or the priority tree, which is what the `unreachable!` arms in
    // `reset` and `save_step` rest on -- between calls a head slot is always `Reset` or `Final`,
    // because no other code puts anything there.

    /// The type of one stored slot.
    #[inline]
    fn slot_type(&self, env: usize, slot: u32) -> SlotType {
        SlotType::from_raw(self.types[[env, slot as usize]])
    }

    #[inline]
    fn set_slot_type(&mut self, env: usize, slot: u32, ty: SlotType) {
        self.types[[env, slot as usize]] = ty as _;
    }

    /// Sets a slot's sampling weight.
    ///
    /// Slots that do not hold a complete transition are given zero, which is what keeps
    /// [`PrioTree::sample`] from ever returning them. Doing so is not merely tidiness: once the
    /// buffer has wrapped, a slot arrives at the head still carrying the priority it was given on
    /// the previous lap, and leaving that in place would make the write head samplable.
    fn set_slot_prio(&mut self, env: usize, slot: u32, prio: f32) {
        let index = env * self.env_capacity() + slot as usize;
        if let Some(prios) = self.prios.as_mut() {
            prios
                .update(index, prio)
                .expect("slot is in range, and every priority reaching here is finite and >= 0");
        }
    }

    /// The type of the slot `env`'s next observation will be written to.
    #[inline]
    fn head_type(&self, env: usize) -> SlotType {
        self.slot_type(env, self.env_heads[env])
    }

    /// The slot written just before `env`'s head, or `None` while the environment is still empty.
    fn prev_written(&self, env: usize) -> Option<u32> {
        let capacity = self.env_capacity() as u32;
        (0 < self.env_lens[env]).then(|| (self.env_heads[env] + capacity - 1) % capacity)
    }

    /// Moves `env`'s head one slot on, wrapping and growing its length.
    fn advance_head(&mut self, env: usize) {
        let capacity = self.env_capacity() as u32;
        self.env_heads[env] = (self.env_heads[env] + 1) % capacity;
        self.env_lens[env] = (self.env_lens[env] + 1).min(capacity);
    }

    /// The slot `age` steps back from `env`'s write head.
    #[inline]
    fn slot_at_age(&self, env: usize, age: u32) -> u32 {
        let capacity = self.env_capacity() as u32;
        (self.env_heads[env] + capacity - age % capacity) % capacity
    }

    /// How far back from `env`'s write head a slot sits.
    #[inline]
    fn slot_age(&self, env: usize, slot: u32) -> u32 {
        let capacity = self.env_capacity() as u32;
        (self.env_heads[env] + capacity - slot) % capacity
    }

    // The write path, and the episode-boundary state machine that runs inside it. Both of these
    // are two passes over the environments on purpose: the first walks the state machine one
    // environment at a time and records the slot that call's incoming observation lands in, and
    // only then does the second write every observation in a single batched copy. `obs_slots`
    // carries the first pass's answer to the second.

    /// Seeds every environment with its initial observation.
    ///
    /// `obs` is indexed by environment -- shape `(num_envs, *obs_shape)`, or
    /// `(num_envs, obs_stack, *obs_shape)` where the buffer has a frame stack, in which case only
    /// the newest frame is kept. It must already be in the dtype the buffer was built with;
    /// nothing is cast on the way in.
    ///
    /// Normally called once, before the first [`save_step`](Self::save_step): autoreset boundaries
    /// during play are inferred from the `terminated` and `truncated` flags rather than from
    /// further `reset` calls. Calling it again mid-run is still well defined, and means what
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
    /// A batch of the wrong shape is an [`InvalidShape`](ReplayBufferError::InvalidShape), raised
    /// before anything is written. The dtype cannot be wrong here: `obs` is a `DynArrayView` of
    /// whatever this buffer stores, which is what it was extracted as.
    pub fn reset(&mut self, obs: DynArrayView) -> ReplayBufferResult<()> {
        let obs = obs_frame("obs", obs, self.observations.shape(), self.obs_stack)?;

        for env in 0..self.num_envs() {
            match self.head_type(env) {
                // Nothing can reach the head slot, so the observation simply lands there.
                SlotType::Reset => {}
                SlotType::Final => {
                    // The head already holds an observation. Where the transition before it is
                    // still mid-episode that observation is its `next_obs`, so overwriting it would
                    // corrupt the pair; an abandoned episode is exactly a truncation, so close the
                    // transition off as one and start a slot later. Otherwise -- a repeated
                    // `reset`, or an episode that has already ended -- nothing reads the head and
                    // it is free.
                    if let Some(prev) = self.prev_written(env)
                        && self.slot_type(env, prev) == SlotType::Normal
                    {
                        self.set_slot_type(env, prev, SlotType::Truncated);
                        self.advance_head(env);
                    }
                }
                ty => unreachable!("head slot is {ty:?}, which no completed call leaves behind"),
            }

            let head = self.env_heads[env];
            self.set_slot_type(env, head, SlotType::Final);
            self.set_slot_prio(env, head, 0.0);

            self.obs_slots[env] = head;
        }
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
    /// An argument of the wrong shape is an [`InvalidShape`](ReplayBufferError::InvalidShape),
    /// raised before anything is written.
    pub fn save_step(
        &mut self,
        actions: DynArrayView,
        rewards: &ArrayRef1<f32>,
        terminated: &ArrayRef1<bool>,
        truncated: &ArrayRef1<bool>,
        next_obs: DynArrayView,
    ) -> ReplayBufferResult<()> {
        check_batch_shape("actions", actions.shape(), self.actions.shape())?;

        let next_obs = obs_frame(
            "next_obs",
            next_obs,
            self.observations.shape(),
            self.obs_stack,
        )?;

        let num_envs = self.num_envs();
        for (name, len) in [
            ("rewards", rewards.len()),
            ("terminated", terminated.len()),
            ("truncated", truncated.len()),
        ] {
            if len != num_envs {
                return Err(ReplayBufferError::invalid_shape(format!(
                    "{name} must have shape [{num_envs}], got [{len}]"
                )));
            }
        }

        // The action and reward complete the transition the head already holds. On an autoreset
        // call the head holds no such transition and these are placeholders, which the next call
        // overwrites in place.
        dyn_write_batch(&mut self.actions, &self.env_heads, &actions);
        write_batch(&mut self.rewards, &self.env_heads, rewards);

        for env in 0..self.num_envs() {
            let (terminated, truncated) = (terminated[env], truncated[env]);

            match self.head_type(env) {
                // The head holds the observation this transition starts from, so the action and
                // reward just written complete it.
                SlotType::Final => {
                    let slot = self.env_heads[env];
                    self.set_slot_type(
                        env,
                        slot,
                        match (terminated, truncated) {
                            (true, _) => SlotType::Terminal,
                            (false, true) => SlotType::Truncated,
                            (false, false) => SlotType::Normal,
                        },
                    );
                    self.set_slot_prio(env, slot, self.max_prio);
                    self.advance_head(env);
                }
                // Gymnasium's next-step autoreset: the episode ended on the previous call, so this
                // one carries the new episode's first observation together with a placeholder
                // action and reward. There is no transition to complete, and the placeholders are
                // overwritten by the next call, which lands on this same slot.
                SlotType::Reset => {}
                ty => unreachable!("head slot is {ty:?}, which no completed call leaves behind"),
            }

            let obs_slot = self.env_heads[env];
            self.set_slot_type(env, obs_slot, SlotType::Final);
            self.set_slot_prio(env, obs_slot, 0.0);

            // Under next-step autoreset the observation just claimed is the episode's *last*, and
            // the reset observation arrives on the following call. Leave the final observation
            // where it is and give the new episode a slot of its own.
            if terminated || truncated {
                self.advance_head(env);
                let head = self.env_heads[env];
                self.set_slot_type(env, head, SlotType::Reset);
                self.set_slot_prio(env, head, 0.0);
            }

            self.obs_slots[env] = obs_slot;
        }
        dyn_write_batch(&mut self.observations, &self.obs_slots, &next_obs);

        Ok(())
    }

    // Drawing a batch: the age band a draw may land on, the frame stack behind it and the
    // n-step rollout in front of it. What these do not cover is *which* transition to draw, which
    // is `sampling`'s job; it reaches them through the crate root.

    /// The ages a draw may land on in `env`, per [`valid_ages`], or `None` when it cannot serve
    /// one yet.
    ///
    /// Recomputed wherever it is needed rather than gathered into a per-environment table up
    /// front: `n_steps` is the only part of it that a draw brings along, so the whole band is
    /// three integer operations on state the buffer already has.
    fn samplable_slot_ages(&self, env: usize, n_steps: u32) -> Option<RangeInclusive<u32>> {
        valid_ages(
            self.env_lens[env],
            self.env_capacity() as _,
            self.obs_stack.map_or(1, NonZero::get),
            n_steps,
        )
    }

    /// Fills `out` with the slots holding the frames of the stack ending at `slot`, oldest first.
    ///
    /// The walk stops at an episode boundary and repeats the oldest frame it reached, which is
    /// what an environment's own frame-stack wrapper does at the start of an episode -- so the
    /// batch matches what the policy actually saw. A boundary is a `Final` slot: those hold an
    /// observation and no action, and every one of them is either the write head, which the age
    /// band keeps this walk away from, or the observation an ended episode was parked on, whose
    /// successor belongs to the next episode.
    fn stack_from(&self, env: usize, mut slot: u32, out: &mut [u32]) {
        let capacity = self.env_capacity() as u32;
        let (newest, rest) = out.split_last_mut().expect("a stack is at least one frame");
        *newest = slot;

        for i in (0..rest.len()).rev() {
            let prev = (slot + capacity - 1) % capacity;
            if self.slot_type(env, prev) == SlotType::Final {
                // The episode starts here. Repeat its opening frame over the rest of the stack,
                // which is what the environment's own frame-stack wrapper does.
                rest[..=i].fill(slot);
                break;
            }
            slot = prev;
            rest[i] = slot;
        }
    }

    /// Rolls an `n_steps` return forward from `slot`, or rejects the draw.
    ///
    /// Returns the discounted return, the slot holding the observation the rollout ended on, and
    /// whether it ended on a true episode end -- so that nothing is bootstrapped past it.
    ///
    /// Rejects when `slot` holds no transition, and when the rollout meets a truncation with steps
    /// still to go: the caller applies one `discount ** n_steps` to the whole batch, so a return
    /// that stopped early would be bootstrapped with the wrong exponent -- unless nothing is
    /// bootstrapped at all, which is exactly the terminal case, and why that one is kept.
    fn rollout(
        &self,
        env: usize,
        slot: u32,
        n_steps: u32,
        discount: f32,
    ) -> Option<(f32, u32, bool)> {
        let capacity = self.env_capacity() as u32;
        let (mut ret, mut discounted, mut slot) = (0.0, 1.0, slot);

        for step in 0..n_steps {
            let ty = self.slot_type(env, slot);
            ret += discounted * self.rewards[[env, slot as usize]];
            discounted *= discount;
            slot = (slot + 1) % capacity;

            match ty {
                SlotType::Normal => {}
                SlotType::Terminal => return Some((ret, slot, true)),
                SlotType::Truncated => return (step + 1 == n_steps).then_some((ret, slot, false)),
                // The write head and the observation parked after an episode end hold no reward.
                // Only a rollout's *first* slot can be one of those: every parked observation sits
                // directly behind a boundary, and the walk stops at every boundary. So this is a
                // draw to reject, not a buffer to distrust.
                SlotType::Final | SlotType::Reset => {
                    debug_assert_eq!(step, 0, "rollout walked into an incomplete slot");
                    return None;
                }
            }
        }

        Some((ret, slot, false))
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
    pub fn sample(
        &mut self,
        batch_size: usize,
        params: &ReplayBufferSampleParams,
    ) -> ReplayBufferResult<Batch> {
        if params.n_steps.get() != 1 && params.discount.is_none() {
            return Err(ReplayBufferError::invalid_argument(
                "discount must be set if n_steps is not one",
            ));
        }
        let discount = params.discount_or_one();

        if !(0.0 < discount && discount <= 1.0) {
            return Err(ReplayBufferError::invalid_argument(
                "discount must be in the range (0, 1]",
            ));
        }

        if batch_size == 0 {
            return Err(ReplayBufferError::invalid_argument(
                "batch_size must be at least one",
            ));
        }

        let capacity = self.env_capacity();
        let batch = sampling::draw_batch(self, batch_size, params)?;

        // `[batch, obs_stack, ..obs_shape]`, with the stack axis only where the buffer has one.
        let mut obs_shape = Vec::with_capacity(self.observations.ndim());
        obs_shape.push(batch_size);
        if let Some(obs_stack) = self.obs_stack {
            obs_shape.push(obs_stack.get() as _);
        }
        obs_shape.extend_from_slice(&self.observations.shape()[2..]);

        // Field order here is the borrow checker's, not `Batch`'s: the action gather borrows
        // `batch.indices` and `Array1::from_vec` moves it, so every gather has to be written
        // before the two `from_vec`s that consume their columns.
        Ok(Batch {
            // `dyn_gather` returns one row per slot, so a stacked draw arrives with its frames
            // folded into the batch axis; `obs_shape` unfolds them. The count is the same either
            // way -- `obs_slots` holds `batch_size * obs_stack` entries -- so the reshape cannot
            // fail, and it is free on a freshly allocated contiguous array.
            obs: dyn_gather(&self.observations, &batch.obs_slots, capacity)
                .into_shape_with_order(obs_shape.clone())
                .expect("the gather returned one row per stacked frame"),
            actions: dyn_gather(&self.actions, &batch.indices, capacity),
            rewards: Array1::from_vec(batch.returns),
            terminals: Array1::from_vec(batch.terminals),
            next_obs: dyn_gather(&self.observations, &batch.next_obs_slots, capacity)
                .into_shape_with_order(obs_shape)
                .expect("the gather returned one row per stacked frame"),
            indices: Array1::from_vec(batch.indices),
            prios: Array1::from_vec(batch.prios),
        })
    }

    // Priorities.

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
    /// A buffer built without priorities has nothing to update, and says so with a
    /// [`RequiresPriorities`](ReplayBufferError::RequiresPriorities).
    pub fn update_prios(
        &mut self,
        indices: &ArrayRef1<u32>,
        prios: &ArrayRef1<f32>,
    ) -> ReplayBufferResult<()> {
        let Some(rb_prios) = &mut self.prios else {
            return Err(ReplayBufferError::requires_priorities(
                "this buffer was built without use_prios, so it keeps no priorities",
            ));
        };
        if indices.len() != prios.len() {
            return Err(ReplayBufferError::invalid_shape(
                "indices and prios must be the same length",
            ));
        }

        // Validate the whole batch before touching the tree. Failing part way through would leave
        // the buffer sampling from a mix of old and new priorities, with nothing to tell the
        // caller how far the update got.
        let len = rb_prios.len();
        for (&i, &p) in indices.iter().zip(prios) {
            if len <= i as usize {
                return Err(ReplayBufferError::index_out_of_range(format!(
                    "index {i} is out of range for a buffer of {len} transitions"
                )));
            }
            if !p.is_finite() || p < 0.0 {
                return Err(ReplayBufferError::invalid_argument(format!(
                    "priority {p} must be finite and non-negative"
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
    /// A buffer built without priorities has no such constant, and says so with a
    /// [`RequiresPriorities`](ReplayBufferError::RequiresPriorities).
    pub fn sum_prios(&self) -> ReplayBufferResult<f32> {
        let Some(prios) = &self.prios else {
            return Err(ReplayBufferError::requires_priorities(
                "this buffer was built without use_prios, so it keeps no priorities",
            ));
        };

        Ok(prios.total())
    }
}

/// Scaffolding shared by this module's tests and `sampling`'s.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    use crate::dyn_array::DType;

    /// The smallest buffer the walks care about: one scalar observation and action per slot, since
    /// nothing below gathers either of them. Seeded, because a draw advances the buffer's own
    /// generator and the assertions below are about which slots come back.
    pub(crate) fn spec() -> ReplayBufferSpec<'static> {
        ReplayBufferSpec::new(&[], DType::U8, &[], DType::U8).with_seed(Some(0))
    }

    pub(crate) fn params(n_steps: u32) -> ReplayBufferSampleParams {
        ReplayBufferSampleParams::new()
            .with_n_steps(NonZero::new(n_steps).unwrap())
            .with_discount(Some(0.99))
    }

    /// A one-environment buffer with its slots laid out directly, rather than driven there through
    /// the write path: `types` is the whole environment, one entry per slot, and each slot's reward
    /// is its own index so that a rollout's arithmetic is readable at a glance.
    pub(crate) fn one_env(
        head: u32,
        len: u32,
        types: &[SlotType],
        spec: &ReplayBufferSpec<'_>,
    ) -> ReplayBuffer {
        let capacity = types.len() as u32;
        let mut rb = ReplayBuffer::new(NonZero::new(1).unwrap(), capacity, spec).unwrap();

        rb.env_heads[0] = head;
        rb.env_lens[0] = len;
        for (slot, &ty) in types.iter().enumerate() {
            rb.types[[0, slot]] = ty as u8;
            rb.rewards[[0, slot]] = slot as f32;
        }

        rb
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

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

    /// [`obs_frame`], with the error kept out of the assertions -- only the shape it picks is
    /// under test here, and every rejection is the same `None` to these cases.
    fn frame(batch: &ArrayD<u8>, stored: &[usize], stack: Option<u32>) -> Option<ArrayD<u8>> {
        let batch = DynArrayView::U8(batch.view());
        let stack = stack.map(|stack| NonZero::new(stack).unwrap());

        match obs_frame("obs", batch, stored, stack) {
            Ok(DynArrayView::U8(frame)) => Some(frame.to_owned()),
            Ok(_) => unreachable!("the batch went in as u8"),
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

    /// A one-environment buffer driven through `reset`/`save_step`. One environment's bookkeeping
    /// *is* the boundary state machine's whole context, so a scalar observation per slot is all
    /// these need -- and every call writes the next value of a counter as its observation, so
    /// `obs` reads back as a map of which call's observation landed in which slot, against the
    /// `types` of the same slots. A slot still holding `0` was never written.
    ///
    /// Reading the observations rather than the slot the write picked is deliberate: the slot is
    /// only `save_step`'s scratch on the way there, and two `Final` slots in the same buffer are
    /// told apart by what is *in* them, not by their type.
    struct TestEnv {
        rb: ReplayBuffer,
        /// The observation the next call writes. Starts at one, so `0` means untouched.
        tick: u8,
    }

    impl TestEnv {
        fn new(capacity: usize) -> Self {
            let spec = ReplayBufferSpec::new(&[], DType::U8, &[], DType::U8).with_use_prios(true);
            let envs = NonZero::new(1).unwrap();
            let rb = ReplayBuffer::new(envs, capacity as u32, &spec).unwrap();

            Self { rb, tick: 1 }
        }

        fn head(&self) -> u32 {
            self.rb.env_heads[0]
        }

        fn len(&self) -> u32 {
            self.rb.env_lens[0]
        }

        /// The one-environment action batch. Nothing here reads an action back.
        fn no_action() -> DynArrayView<'static> {
            DynArrayView::U8(aview1(&[0u8]).into_dyn())
        }

        fn reset(&mut self) {
            let obs = [self.tick];
            (self.rb)
                .reset(DynArrayView::U8(aview1(&obs).into_dyn()))
                .expect("sizes should match");
            self.tick += 1;
        }

        fn step(&mut self, terminated: bool, truncated: bool) {
            let next_obs = [self.tick];
            (self.rb)
                .save_step(
                    Self::no_action(),
                    &aview1(&[0.0f32]),
                    &aview1(&[terminated]),
                    &aview1(&[truncated]),
                    DynArrayView::U8(aview1(&next_obs).into_dyn()),
                )
                .expect("sizes should match");
            self.tick += 1;
        }

        /// The stored observation of every slot, in slot order.
        fn obs(&self) -> Vec<u8> {
            match &self.rb.observations {
                DynArray::U8(obs) => obs.index_axis(Axis(0), 0).iter().copied().collect(),
                other => unreachable!("built as u8, not {:?}", other.dtype()),
            }
        }

        fn types(&self) -> Vec<SlotType> {
            (self.rb.types)
                .iter()
                .copied()
                .map(SlotType::from_raw)
                .collect()
        }

        fn prios(&self) -> &[f32] {
            (self.rb.prios)
                .as_ref()
                .expect("built with priorities")
                .prios()
        }
    }

    use SlotType::{Final, Normal, Reset, Terminal, Truncated};

    #[test]
    fn test_reset_then_steps_fill_consecutive_slots() {
        let mut env = TestEnv::new(6);
        env.reset();
        assert_eq!((env.head(), env.len()), (0, 0));

        for _ in 0..3 {
            env.step(false, false);
        }

        // Each `next_obs` lands one slot on from the transition the step just completed, so the
        // four observations written so far sit in the first four slots in order.
        assert_eq!(env.obs(), [1, 2, 3, 4, 0, 0]);
        assert_eq!(env.types(), [Normal, Normal, Normal, Final, Reset, Reset]);
        assert_eq!((env.head(), env.len()), (3, 3));
    }

    #[test]
    fn test_repeated_reset_spends_no_slot() {
        // Nothing can read the head observation yet, so a second and third `reset` just overwrite
        // it. This used to fall through the state machine and panic.
        let mut env = TestEnv::new(4);
        for tick in 1..=3 {
            env.reset();
            // Overwritten in place rather than parked in a slot of its own.
            assert_eq!(env.obs(), [tick, 0, 0, 0]);
            assert_eq!((env.head(), env.len()), (0, 0));
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
        // Both slot 2 and slot 3 are `Final`; only the observations say which is the abandoned
        // episode's last and which is the new episode's first.
        assert_eq!(env.obs(), [1, 2, 3, 4, 0, 0]);
        assert_eq!(env.types(), [Normal, Truncated, Final, Final, Reset, Reset]);
        assert_eq!((env.head(), env.len()), (3, 3));
    }

    #[test]
    fn test_reset_after_an_episode_end_spends_no_slot() {
        let mut env = TestEnv::new(6);
        env.reset();
        env.step(true, false);
        assert_eq!(env.types(), [Terminal, Final, Reset, Reset, Reset, Reset]);
        let (head, len) = (env.head(), env.len());

        // The head is the empty slot the termination left behind, so the reset observation lands
        // straight in it.
        env.reset();
        assert_eq!(env.obs(), [1, 2, 3, 0, 0, 0]);
        assert_eq!(env.types(), [Terminal, Final, Final, Reset, Reset, Reset]);
        assert_eq!((env.head(), env.len()), (head, len));
    }

    #[test]
    fn test_terminated_step_keeps_the_final_observation() {
        let mut env = TestEnv::new(6);
        env.reset();
        env.step(false, false);

        // Slot 2 keeps the episode's final observation -- it is slot 1's `next_obs` -- and slot 3
        // is left empty for the reset observation the next call brings.
        env.step(true, false);
        assert_eq!(env.obs(), [1, 2, 3, 0, 0, 0]);
        assert_eq!(env.types(), [Normal, Terminal, Final, Reset, Reset, Reset]);
        assert_eq!((env.head(), env.len()), (3, 3));

        // That next call carries a placeholder action and reward and completes no transition, so
        // its observation claims slot 3 and the head does not move.
        env.step(false, false);
        assert_eq!(env.obs(), [1, 2, 3, 4, 0, 0]);
        assert_eq!(env.types(), [Normal, Terminal, Final, Final, Reset, Reset]);
        assert_eq!((env.head(), env.len()), (3, 3));

        env.step(false, false);
        assert_eq!(env.obs(), [1, 2, 3, 4, 5, 0]);
        assert_eq!(env.types(), [Normal, Terminal, Final, Normal, Final, Reset]);
    }

    #[test]
    fn test_truncated_step_matches_terminated_but_for_the_slot_type() {
        let mut env = TestEnv::new(6);
        env.reset();
        env.step(false, true);
        assert_eq!(env.obs(), [1, 2, 0, 0, 0, 0]);
        assert_eq!(env.types(), [Truncated, Final, Reset, Reset, Reset, Reset]);
        assert_eq!((env.head(), env.len()), (2, 2));
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

        let samplable: Vec<bool> = env.prios().iter().map(|&p| p > 0.0).collect();
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
            let head = env.head() as usize;
            assert_eq!(env.prios()[head], 0.0, "head slot {head} is samplable");
            assert!(env.types()[head] == Final);
        }
        assert_eq!(env.len(), 4);

        // Three complete transitions and the head, all lap after lap.
        let samplable = env.prios().iter().filter(|&&p| p > 0.0).count();
        assert_eq!(samplable, 3);
    }

    #[test]
    fn test_valid_ages_leaves_room_for_the_stack_and_the_rollout() {
        // Eight slots written of sixteen, so nine observations exist, at ages 0..=8.
        // A four-frame stack ending at age 5 reads ages 5..=8, which is the oldest a draw can be.
        assert_eq!(valid_ages(8, 16, 4, 1), Some(1..=5));
        // A three-step rollout from age 3 reads ages 3, 2, 1 and lands its `next_obs` on the head.
        assert_eq!(valid_ages(8, 16, 4, 3), Some(3..=5));

        // Full: age 15 is the oldest slot, and its stack would wrap onto the head.
        assert_eq!(valid_ages(16, 16, 4, 1), Some(1..=12));
        assert_eq!(valid_ages(16, 16, 1, 1), Some(1..=15));

        // Nothing to draw yet: not enough frames for a stack, then not enough for the rollout.
        assert_eq!(valid_ages(2, 16, 4, 1), None);
        assert_eq!(valid_ages(4, 16, 4, 3), None);
        assert_eq!(valid_ages(0, 16, 1, 1), None);
    }

    #[test]
    fn test_ages_count_back_from_the_head_and_wrap() {
        let rb = one_env(1, 3, &[Normal, Final, Normal, Normal], &spec());

        assert_eq!(rb.slot_at_age(0, 0), 1);
        assert_eq!(rb.slot_at_age(0, 1), 0);
        assert_eq!(rb.slot_at_age(0, 2), 3);
        assert_eq!(rb.slot_at_age(0, 3), 2);
        assert!((0..4).all(|age| rb.slot_age(0, rb.slot_at_age(0, age)) == age));

        assert_eq!(rb.samplable_slot_ages(0, 1), Some(1..=3));
        assert_eq!(rb.slot_type(0, 1), Final);
    }

    #[test]
    fn test_stack_from_walks_back_and_stops_at_an_episode_boundary() {
        // Slot 2 is the observation a terminated episode was parked on, so slot 3 begins a new
        // episode and nothing before slot 3 belongs to the same stack.
        let types = [Normal, Terminal, Final, Normal, Normal, Normal, Final];
        let rb = one_env(6, 6, &types, &spec().with_obs_stack(NonZero::new(4)));

        let mut out = [0; 4];

        // Three frames into the new episode: the walk reaches slot 3 and repeats it.
        rb.stack_from(0, 5, &mut out);
        assert_eq!(out, [3, 3, 4, 5]);

        rb.stack_from(0, 4, &mut out);
        assert_eq!(out, [3, 3, 3, 4]);

        // The first observation of the new episode has nothing behind it at all.
        rb.stack_from(0, 3, &mut out);
        assert_eq!(out, [3, 3, 3, 3]);

        // The parked observation itself still belongs to the *old* episode, so the walk crosses
        // the terminal into it, and only stops at the write head on slot 6.
        rb.stack_from(0, 2, &mut out);
        assert_eq!(out, [0, 0, 1, 2]);

        // A one-frame stack is just the slot.
        let mut one = [0; 1];
        rb.stack_from(0, 4, &mut one);
        assert_eq!(one, [4]);
    }

    #[test]
    fn test_rollout_accumulates_until_a_boundary() {
        // Rewards are the slot index. Slot 3 ends the episode; slot 4 is the parked observation.
        let types = [Normal, Normal, Normal, Terminal, Final, Normal, Final];
        let rb = one_env(6, 6, &types, &spec());

        // A one-step return is just the slot's own reward, and `next_obs` is the slot after it.
        assert_eq!(rb.rollout(0, 1, 1, 0.5), Some((1.0, 2, false)));

        // Slots 0, 1 and 2 are all `Normal`, so a three-step rollout from slot 0 runs its full
        // length: 0 + 0.5*1 + 0.25*2 = 1.0, ending its `next_obs` on slot 3.
        assert_eq!(rb.rollout(0, 0, 3, 0.5), Some((1.0, 3, false)));

        // Reaching the terminal stops the rollout early, and the bootstrap is masked out.
        let expected = 1.0 + 0.5 * 2.0 + 0.25 * 3.0;
        assert_eq!(rb.rollout(0, 1, 4, 0.5), Some((expected, 4, true)));

        // A rollout may not start on a slot that holds no transition.
        assert_eq!(rb.rollout(0, 4, 1, 0.5), None);
        assert_eq!(rb.rollout(0, 6, 1, 0.5), None);
    }

    #[test]
    fn test_rollout_rejects_a_truncation_it_cannot_reach_the_end_of() {
        use SlotType::Truncated;

        let types = [Normal, Normal, Truncated, Final, Normal, Final];
        let rb = one_env(5, 5, &types, &spec());

        // Landing on the truncation with the rollout's last step is fine: the caller's
        // `discount ** n_steps` is the right exponent, and `next_obs` is the successor to
        // bootstrap from.
        assert_eq!(rb.rollout(0, 1, 2, 0.5), Some((1.0 + 0.5 * 2.0, 3, false)));

        // Reaching it with steps still to go is not: the return would be short but bootstrapped
        // as if it were not.
        assert_eq!(rb.rollout(0, 1, 3, 0.5), None);
        assert_eq!(rb.rollout(0, 2, 2, 0.5), None);
    }

    #[test]
    fn test_spec_max_prio_is_the_scale_new_transitions_are_written_at() {
        // What a resumed run needs: its refilled buffer writes at the scale training had reached,
        // not at a fresh buffer's 1.0.
        let spec = spec()
            .with_use_prios(true)
            .with_max_prio(0.25)
            .with_max_prio_decay(0.5);
        let mut rb = ReplayBuffer::new(NonZero::new(1).unwrap(), 8, &spec).unwrap();
        assert_eq!(rb.max_prio(), 0.25);

        let obs = ArrayD::<u8>::zeros(IxDyn(&[1]));
        let step = |rb: &mut ReplayBuffer| {
            rb.save_step(
                DynArrayView::U8(obs.view()),
                &array![0.0],
                &array![false],
                &array![false],
                DynArrayView::U8(obs.view()),
            )
            .unwrap()
        };

        // Slot 0 becomes a whole transition once slot 1 holds its `next_obs`, and is given
        // `max_prio` as it does.
        rb.reset(DynArrayView::U8(obs.view())).unwrap();
        step(&mut rb);
        step(&mut rb);
        assert_eq!(rb.prios.as_ref().unwrap().prios()[0], 0.25);

        // And the decay runs from there rather than from 1.0.
        rb.update_prios(&array![0], &array![0.1]).unwrap();
        assert_eq!(rb.max_prio(), 0.125);
    }

    #[test]
    fn test_spec_max_prio_takes_what_a_priority_takes() {
        let build = |max_prio| {
            ReplayBuffer::new(NonZero::new(1).unwrap(), 8, &spec().with_max_prio(max_prio))
        };

        // Zero is reachable by decay, so it is reachable here: new transitions simply stay
        // unsamplable until something writes a priority.
        assert_eq!(build(0.0).unwrap().max_prio(), 0.0);

        assert!(build(-1.0).is_err());
        assert!(build(f32::NAN).is_err());
        assert!(build(f32::INFINITY).is_err());
    }
}
