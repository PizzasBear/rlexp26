//! The Python boundary: the `expreplay` module and [`PyReplayBuffer`], a thin wrapper that
//! converts arrays, dtypes and errors and decides nothing itself.
//!
//! The pymethods' doc comments become Python `__doc__`s, so they are written for a Python reader,
//! with no rustdoc links. `expreplay.pyi` carries the fuller version.

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
            sampling::DrawError::TooSmall | sampling::DrawError::NoPriorities => {
                PyValueError::new_err(value.to_string())
            }
            // The buffer's own state is degenerate, not the caller's arguments.
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

/// How an argument of one dtype arrives from Python. [`PyArrayLikeDyn`] borrows an array of
/// exactly this dtype and otherwise rebuilds the argument element by element through `Vec<T>`,
/// which pyo3 cannot do for `f16` or `bf16`; those take a real array of their own dtype only.
macro_rules! dyn_py_arg {
    (BF16, $lt:lifetime, $ty:ty) => { numpy::PyReadonlyArrayDyn<$lt, $ty> };
    (F16, $lt:lifetime, $ty:ty) => { numpy::PyReadonlyArrayDyn<$lt, $ty> };
    ($variant:ident, $lt:lifetime, $ty:ty) => { PyArrayLikeDyn<$lt, $ty> };
}

/// The `TypeError` for an argument of the wrong dtype, naming the argument, the stored dtype and
/// what arrived. pyo3's own reads `'ndarray' object is not an instance of 'ndarray'`.
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
        /// A NumPy argument extracted as whichever dtype the buffer stores; see [`dyn_py_arg`].
        pub enum DynPyArrayLike<'py> {
            $($variant(dyn_py_arg!($variant, 'py, $ty)),)+
        }

        impl<'py> DynPyArrayLike<'py> {
            /// Extracts `obj` as `dtype`, raising `TypeError` if it cannot be read as one. `name` is
            /// the argument's Python name, as the core's shape errors use it.
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
    /// ``obs_shape`` and ``act_shape`` describe one observation and action. Either dtype may be
    /// uint8/16/32/64, int8/16/32/64, float16, bfloat16, float32 or float64; they default to
    /// float32 observations and uint8 actions.
    ///
    /// ``obs_stack`` makes ``sample`` return that many frames per observation and lets
    /// ``save_step`` take a stacked one. ``use_prios`` enables prioritised sampling, and
    /// ``stratified`` (defaulting to ``use_prios``, which it requires) spreads a batch over the
    /// priority mass. ``max_prio`` seeds the priority new transitions are given.
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

        // Unset rather than defaulted, so the defaults live only in `ReplayBufferSpec::new`.
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

        // Zeroing tens of gigabytes touches no Python object.
        let buf = py.detach(|| ReplayBuffer::new(num_envs, env_capacity, &spec))?;

        Ok(Self(buf))
    }

    /// Number of transitions written, summed across environments. Includes the unsamplable slots
    /// around each write head, so it is more than ``sample`` can draw.
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
    /// independently. Settable; raises ValueError on a buffer without ``use_prios``.
    #[getter]
    fn stratified(&self) -> bool {
        self.0.stratified()
    }

    #[setter]
    fn set_stratified(&mut self, stratified: bool) -> PyResult<()> {
        Ok(self.0.set_stratified(stratified)?)
    }

    /// The priority a newly written transition is given, and the scale ``sample``'s priorities are
    /// reported against. Read-only.
    #[getter]
    fn max_prio(&self) -> f32 {
        self.0.max_prio()
    }

    /// How fast ``max_prio`` decays, once per ``update_prios`` call, so its half-life is in
    /// gradient steps. Settable.
    #[getter]
    fn max_prio_decay(&self) -> f32 {
        self.0.max_prio_decay()
    }

    #[setter]
    fn set_max_prio_decay(&mut self, max_prio_decay: f32) -> PyResult<()> {
        Ok(self.0.set_max_prio_decay(max_prio_decay)?)
    }

    /// Seed every environment with its initial observation, ``(num_envs, *obs_shape)`` or
    /// ``(num_envs, obs_stack, *obs_shape)``, in the buffer's dtype. Called mid-run it abandons the
    /// current episode, at the cost of at most one slot.
    ///
    /// Raises TypeError on a dtype mismatch and ValueError on a shape mismatch, before anything is
    /// written.
    fn reset(&mut self, obs: &Bound<'_, PyAny>) -> PyResult<()> {
        let obs = DynPyArrayLike::extract("obs", self.0.obs_dtype(), obs)?;

        Ok(self.0.reset(obs.as_array())?)
    }

    /// Record one transition per environment and advance each write head. ``next_obs`` takes the
    /// shapes ``reset``'s ``obs`` does. Under ``AutoresetMode.NEXT_STEP`` the call after an episode
    /// ends completes no transition.
    ///
    /// Raises TypeError on a dtype mismatch and ValueError on a shape mismatch, before anything is
    /// written.
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
    /// ``indices`` go back to ``update_prios``. ``prios`` are the stored priorities, unnormalised,
    /// and 1.0 throughout without priorities. ``rewards`` is the ``n_steps`` return and
    /// ``next_obs`` the observation ``n_steps`` later; ``terminals`` marks true episode ends only.
    ///
    /// Raises ValueError if ``discount`` is missing or out of range, ``batch_size`` is zero, or
    /// nothing can be drawn yet, and RuntimeError if the priorities have collapsed onto slots that
    /// cannot be drawn.
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

        // The draw touches no Python object, and the caller may have a device transfer in flight.
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

    /// Replace the priorities at ``indices`` with ``prios``, which must be finite and
    /// non-negative; the last value for a duplicate index wins. No exponent is applied. Also
    /// refreshes ``max_prio``.
    ///
    /// Raises ValueError without priorities or on an invalid priority, and IndexError on an index
    /// out of range.
    fn update_prios(
        &mut self,
        indices: PyArrayLike1<u32>,
        prios: PyArrayLike1<f32>,
    ) -> PyResult<()> {
        Ok(self
            .0
            .update_prios(&indices.as_array(), &prios.as_array())?)
    }

    /// Sum of every stored priority, which turns ``sample``'s priorities into probabilities.
    ///
    /// Raises ValueError if this buffer was built without priorities.
    #[getter]
    fn sum_prios(&self) -> PyResult<f32> {
        Ok(self.0.sum_prios()?)
    }
}
