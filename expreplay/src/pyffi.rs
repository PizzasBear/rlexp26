//! The Python boundary: the `expreplay` extension module and everything that exists only to
//! serve it.
//!
//! [`PyReplayBuffer`] is a thin wrapper over [`ReplayBuffer`]. Its job is to convert -- NumPy
//! arrays into `ndarray` views, dtypes into [`DType`], a [`ReplayBufferError`] into the exception
//! a Python caller expects -- and then to get out of the way. Nothing here decides anything about
//! how the buffer behaves; if a rule needs enforcing it belongs in the core, and so does the
//! reasoning behind it.
//!
//! The doc comments below are the exception, because pyo3 turns them into each method's Python
//! `__doc__`: they are what `help(ReplayBuffer)` prints. So they are written for a Python reader
//! -- plain prose, no rustdoc link syntax, no `Self::` paths -- and kept to what that reader
//! needs at the prompt. The full account of *why* each rule holds stays on [`ReplayBuffer`]'s own
//! methods, and `expreplay.pyi` carries the version a type checker and an IDE read.

use std::num::NonZero;

use numpy::{
    PyArray1, PyArrayDescr, PyArrayLike1, PyArrayLikeDyn, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{
        PyIndexError, PyMemoryError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError,
    },
    prelude::*,
    types::PyTuple,
};

use crate::dyn_array::{DType, DynArrayView, for_all_dtypes};
use crate::utils::AllocationError;
use crate::{
    ReplayBuffer, ReplayBufferError, ReplayBufferSampleParams, ReplayBufferSpec, sampling,
};

#[pymodule]
mod expreplay {
    #[pymodule_export]
    use super::PyReplayBuffer;
}

impl From<AllocationError> for PyErr {
    fn from(value: AllocationError) -> Self {
        PyMemoryError::new_err(value.to_string())
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

impl From<ReplayBufferError> for PyErr {
    fn from(value: ReplayBufferError) -> Self {
        match value {
            ReplayBufferError::AllocationError(err) => err.into(),
            ReplayBufferError::Draw(err) => err.into(),
            ReplayBufferError::InvalidArgument(msg)
            | ReplayBufferError::InvalidShape(msg)
            | ReplayBufferError::RequiresPriorities(msg) => PyValueError::new_err(msg),
            ReplayBufferError::ArgumentOverflow(msg) => PyOverflowError::new_err(msg),
            ReplayBufferError::IndexOutOfRange(msg) => PyIndexError::new_err(msg),
        }
    }
}

/// How an argument of one dtype arrives from Python.
///
/// [`PyArrayLikeDyn`] takes a NumPy array of exactly this dtype as a borrow, and falls back to
/// rebuilding anything else NumPy can read -- a list, a JAX array -- element by element through
/// the sequence protocol. That fallback goes via `Vec<T>`, which pyo3 cannot produce for
/// [`half::f16`]: there is no conversion from a Python float to one. `f16` therefore takes a real
/// NumPy `float16` array and nothing else, which is what the caller should be passing anyway.
///
/// Both types expose `as_array`, so only the type named here differs between the two paths.
macro_rules! dyn_py_arg {
    (F16, $lt:lifetime, $ty:ty) => { numpy::PyReadonlyArrayDyn<$lt, $ty> };
    ($variant:ident, $lt:lifetime, $ty:ty) => { PyArrayLikeDyn<$lt, $ty> };
}

/// The `TypeError` an argument of the wrong dtype gets.
///
/// pyo3's own reads `'ndarray' object is not an instance of 'ndarray'` -- accurate, in that the
/// array handed over is not an array of the stored dtype, and unreadable. Name the argument, what
/// the buffer stores and what turned up instead, which for a `float16` buffer also explains the
/// one dtype that refuses a list; see [`dyn_py_arg`].
fn wrong_dtype(name: &str, dtype: DType, obj: &Bound<'_, PyAny>) -> PyErr {
    let py = obj.py();
    let got = match obj.cast::<PyUntypedArray>() {
        Ok(array) => format!("a {} array", array.dtype()),
        Err(_) => match obj.get_type().name() {
            Ok(ty) => format!("a {ty}"),
            Err(_) => "something else".to_owned(),
        },
    };

    PyTypeError::new_err(format!(
        "{name} must be a {} array, got {got}; nothing is cast on the way in",
        dtype.np(py),
    ))
}

macro_rules! def_dyn_py_array_like {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        /// A NumPy argument whose dtype is only known at runtime, extracted as whichever one the
        /// buffer stores. See [`dyn_py_arg`] for what each variant will accept.
        pub enum DynPyArrayLike<'py> {
            $($variant(dyn_py_arg!($variant, 'py, $ty)),)+
        }

        impl<'py> DynPyArrayLike<'py> {
            /// Extracts `obj` as `dtype`, raising `TypeError` if it cannot be read as one.
            ///
            /// `name` is the argument's Python name, for the error. It matches the one the core's
            /// shape errors use, so a caller who got the dtype wrong and a caller who got the
            /// shape wrong are told about the same argument by the same name.
            pub fn extract(name: &str, dtype: DType, obj: &Bound<'py, PyAny>) -> PyResult<Self> {
                Ok(match dtype {
                    $(DType::$variant => Self::$variant(
                        obj.extract().map_err(|_| wrong_dtype(name, dtype, obj))?,
                    ),)+
                })
            }

            pub fn as_array(&self) -> DynArrayView<'_> {
                match self {
                    $(Self::$variant(bound) => DynArrayView::$variant(bound.as_array()),)+
                }
            }
        }
    };
}
for_all_dtypes!(def_dyn_py_array_like);

/// What [`PyReplayBuffer::sample`] returns: `(indices, prios, obs, actions, rewards, terminals,
/// next_obs)`.
type PyBatch<'py> = (
    Bound<'py, PyArray1<u32>>,  // indices
    Bound<'py, PyArray1<f32>>,  // priorities
    Bound<'py, PyAny>,          // observations
    Bound<'py, PyAny>,          // actions
    Bound<'py, PyArray1<f32>>,  // rewards
    Bound<'py, PyArray1<bool>>, // terminals
    Bound<'py, PyAny>,          // next observations
);

#[pyclass(name = "ReplayBuffer")]
pub struct PyReplayBuffer(ReplayBuffer);

#[pymethods]
impl PyReplayBuffer {
    /// Allocate a buffer holding ``num_envs * env_capacity`` transitions.
    ///
    /// ``obs_shape`` and ``act_shape`` describe a single observation and action; the environment
    /// and slot axes are prepended internally. Both dtypes accept any of uint8/16/32/64,
    /// int8/16/32/64, float16/32/64, and default to float32 for observations and uint8 for
    /// actions.
    ///
    /// ``obs_stack`` makes ``sample`` return that many consecutive frames per observation and
    /// lets ``save_step`` accept a stacked observation directly; only one frame per slot is
    /// stored either way. ``use_prios`` enables prioritised sampling, and ``stratified``
    /// (defaulting to ``use_prios``, and requiring it) spreads a batch over the priority mass.
    /// ``max_prio`` seeds the priority new transitions are given, for a run picking up where a
    /// checkpointed one left off; it is read-only afterwards. n-step returns are asked for per
    /// draw, in ``sample``, not here.
    ///
    /// Raises MemoryError if the buffer does not fit, OverflowError if its flat index would not
    /// fit in a uint32, and ValueError for out-of-range parameters.
    #[new]
    #[pyo3(signature = (
        num_envs, env_capacity,
        *,
        obs_shape, obs_dtype = None,
        act_shape = vec![], act_dtype = None,
        obs_stack = None,
        use_prios = false, max_prio = None, max_prio_decay = None, stratified = None,
        seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new<'py>(
        py: Python<'py>,
        num_envs: u32,
        env_capacity: u32,
        obs_shape: Vec<usize>,
        obs_dtype: Option<&Bound<'py, PyAny>>,
        act_shape: Vec<usize>,
        act_dtype: Option<&Bound<'py, PyAny>>,
        obs_stack: Option<NonZero<u32>>,
        use_prios: bool,
        max_prio: Option<f32>,
        max_prio_decay: Option<f32>,
        stratified: Option<bool>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let Some(num_envs) = NonZero::new(num_envs) else {
            return Err(PyValueError::new_err("num_envs must be at least one"));
        };

        let obs_dtype = obs_dtype.map_or(Ok(DType::F32), DType::from_np)?;
        let act_dtype = act_dtype.map_or(Ok(DType::U8), DType::from_np)?;

        // `max_prio` and `max_prio_decay` are left unset rather than defaulted here, so that the
        // numbers themselves live in `ReplayBufferSpec::new` and nowhere else.
        let mut spec = ReplayBufferSpec::new(&obs_shape, obs_dtype, &act_shape, act_dtype)
            .with_obs_stack(obs_stack)
            .with_use_prios(use_prios)
            .with_stratified(stratified)
            .with_seed(seed);
        if let Some(max_prio) = max_prio {
            spec = spec.with_max_prio(max_prio);
        }
        if let Some(max_prio_decay) = max_prio_decay {
            spec = spec.with_max_prio_decay(max_prio_decay);
        }

        // The observation array alone is routinely tens of gigabytes, and zeroing it touches no
        // Python object, so the allocation runs with the GIL released.
        let buf = py.detach(|| ReplayBuffer::new(num_envs, env_capacity, &spec))?;

        Ok(Self(buf))
    }

    /// Number of transitions stored, summed across environments.
    ///
    /// Counts everything written, which is more than ``sample`` can draw: the slots around each
    /// environment's write head are excluded from sampling but included here.
    fn __len__(&self) -> usize {
        self.0.len()
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.0.num_envs()
    }

    #[getter]
    fn env_capacity(&self) -> usize {
        self.0.env_capacity()
    }

    #[getter]
    fn total_capacity(&self) -> usize {
        self.0.total_capacity()
    }

    #[getter]
    fn obs_dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        self.0.obs_dtype().np(py)
    }

    #[getter]
    fn obs_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.obs_shape())
    }

    #[getter]
    fn act_dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        self.0.act_dtype().np(py)
    }

    #[getter]
    fn act_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.act_shape())
    }

    /// Frames per observation that ``sample`` returns, or None if this buffer has no stack.
    #[getter]
    fn obs_stack(&self) -> Option<u32> {
        self.0.obs_stack()
    }

    #[getter]
    fn use_prios(&self) -> bool {
        self.0.use_prios()
    }

    /// Whether a prioritised batch is spread over the priority mass rather than drawn
    /// independently. Settable; changes only how the draws correlate, never what is stored.
    ///
    /// Setting it on a buffer built without ``use_prios`` raises ValueError.
    #[getter]
    fn stratified(&self) -> bool {
        self.0.stratified()
    }

    #[setter]
    fn set_stratified(&mut self, stratified: bool) -> PyResult<()> {
        Ok(self.0.set_stratified(stratified)?)
    }

    /// The priority a newly written transition is given; see ``update_prios``.
    ///
    /// This is the scale ``sample`` reports its priorities against, so an importance-sampling
    /// weight normalised against the whole buffer rather than one batch divides by this.
    ///
    /// Its starting value is a constructor argument, so that a resumed run does not hand its
    /// refilled buffer the priorities of a fresh one, but it is not settable afterwards: every
    /// priority already stored was written against the value in force at the time.
    #[getter]
    fn max_prio(&self) -> f32 {
        self.0.max_prio()
    }

    /// How fast ``max_prio`` decays, applied once per ``update_prios`` call -- so its half-life
    /// is measured in gradient steps, not environment steps. Settable, for a schedule that has to
    /// track a changing replay ratio.
    #[getter]
    fn max_prio_decay(&self) -> f32 {
        self.0.max_prio_decay()
    }

    #[setter]
    fn set_max_prio_decay(&mut self, max_prio_decay: f32) -> PyResult<()> {
        Ok(self.0.set_max_prio_decay(max_prio_decay)?)
    }

    /// Seed every environment with its initial observation.
    ///
    /// ``obs`` is indexed by environment: ``(num_envs, *obs_shape)``, or
    /// ``(num_envs, obs_stack, *obs_shape)`` where the buffer has a frame stack, of which only
    /// the newest frame is kept. It must already be the buffer's dtype; nothing is cast.
    ///
    /// Normally called once, before the first ``save_step`` -- autoreset boundaries during play
    /// are inferred from the ``terminated`` and ``truncated`` flags, not from further ``reset``
    /// calls. Calling it again mid-run means what ``gym.Env.reset`` means: abandon the current
    /// episode and start a new one, which costs at most one slot.
    ///
    /// Raises TypeError on a dtype mismatch and ValueError on a shape mismatch, in both cases
    /// before anything is written.
    fn reset(&mut self, obs: &Bound<'_, PyAny>) -> PyResult<()> {
        let obs = DynPyArrayLike::extract("obs", self.0.obs_dtype(), obs)?;

        Ok(self.0.reset(obs.as_array())?)
    }

    /// Record one transition per environment and advance each write head.
    ///
    /// Every argument is indexed by environment, and ``next_obs`` takes the same two shapes
    /// ``obs`` does in ``reset``. Under Gymnasium's default ``AutoresetMode.NEXT_STEP`` the step
    /// that ends an episode carries that episode's final observation, and the call after it
    /// carries the new episode's first observation with an action and reward the environment
    /// ignored -- that call completes no transition and does not grow ``len(self)``.
    ///
    /// Raises TypeError on a dtype mismatch and ValueError on a shape mismatch, in both cases
    /// before anything is written.
    fn save_step(
        &mut self,
        actions: &Bound<'_, PyAny>,
        rewards: PyArrayLike1<f32>,
        terminated: PyArrayLike1<bool>,
        truncated: PyArrayLike1<bool>,
        next_obs: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let actions = DynPyArrayLike::extract("actions", self.0.act_dtype(), actions)?;
        let next_obs = DynPyArrayLike::extract("next_obs", self.0.obs_dtype(), next_obs)?;

        Ok(self.0.save_step(
            actions.as_array(),
            &rewards.as_array(),
            &terminated.as_array(),
            &truncated.as_array(),
            next_obs.as_array(),
        )?)
    }

    /// Draw ``batch_size`` transitions as
    /// ``(indices, prios, obs, actions, rewards, terminals, next_obs)``.
    ///
    /// ``indices`` are flat ``env * env_capacity + slot`` indices, to hand back to
    /// ``update_prios``. ``prios`` holds each drawn transition's stored priority, raw rather than
    /// normalised so the caller picks its own denominator -- ``sum_prios`` for the sampling
    /// probability, ``max_prio`` for a buffer-wide scale -- and is 1.0 throughout for a buffer
    /// sampling uniformly.
    ///
    /// ``rewards`` is the accumulated ``n_steps`` return and ``next_obs`` the observation
    /// ``n_steps`` later. ``terminals`` marks true episode ends only: a transition cut short by a
    /// time limit should still be bootstrapped from, so it is not marked here.
    ///
    /// Raises ValueError if no environment holds enough transitions yet, or if ``n_steps`` is
    /// greater than one without a ``discount``.
    #[pyo3(signature = (
        batch_size,
        *,
        n_steps = NonZero::<u32>::MIN,
        discount = None,
    ))]
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: usize,
        n_steps: NonZero<u32>,
        discount: Option<f32>,
    ) -> PyResult<PyBatch<'py>> {
        let params = ReplayBufferSampleParams::new()
            .with_n_steps(n_steps)
            .with_discount(discount);

        // The draw walks the buffer and memcpys a whole batch of stacked observations out of it,
        // touching no Python object on the way. That is by far the most time this extension spends
        // in one call, and the loop calling it has a device transfer in flight, so it runs with
        // the GIL released. Only the NumPy arrays below are built back under it.
        let batch = py.detach(|| self.0.sample(batch_size, &params))?;

        Ok((
            PyArray1::from_owned_array(py, batch.indices),
            PyArray1::from_owned_array(py, batch.prios),
            batch.obs.into_np(py),
            batch.actions.into_np(py),
            PyArray1::from_owned_array(py, batch.rewards),
            PyArray1::from_owned_array(py, batch.terminals),
            batch.next_obs.into_np(py),
        ))
    }

    /// Replace the priorities at ``indices`` with ``prios``.
    ///
    /// Both arrays must be the same length, and every priority must be finite and non-negative.
    /// Duplicate indices are applied in order, so the last value for an index wins. No exponent
    /// is applied here: a caller wanting the usual ``p ** alpha`` weighting applies ``alpha``
    /// before calling, since the tree samples in proportion to whatever it stores.
    ///
    /// This also refreshes ``max_prio`` -- the priority newly written transitions are given -- as
    /// the larger of the highest priority in ``prios`` and the previous value decayed by
    /// ``max_prio_decay``.
    ///
    /// Raises ValueError if this buffer was built without priorities.
    fn update_prios(
        &mut self,
        indices: PyArrayLike1<u32>,
        prios: PyArrayLike1<f32>,
    ) -> PyResult<()> {
        Ok(self
            .0
            .update_prios(&indices.as_array(), &prios.as_array())?)
    }

    /// Sum of every stored priority: the constant that turns ``sample``'s priorities into
    /// sampling probabilities.
    ///
    /// Raises ValueError if this buffer was built without priorities.
    #[getter]
    fn sum_prios(&self) -> PyResult<f32> {
        Ok(self.0.sum_prios()?)
    }
}
