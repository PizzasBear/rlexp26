//! The Python boundary: the `expreplay` module, wrapping the core one type and one method at a
//! time. It converts arrays, shapes and errors and enforces no rule of its own.
//!
//! A batch of handles is an `int64` array whose last axis is `(env, t)`, which is [`Handle`]'s
//! layout, and every result takes the leading shape of the handles it answers. The pymethods'
//! doc comments become Python `__doc__`s; `expreplay.pyi` carries the fuller version.

use std::borrow::Cow;

use numpy::{
    IntoPyArray, PyArray1, PyArrayDescr, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray1, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyIndexError, PyMemoryError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    sync::PyOnceLock,
    types::PyDict,
};

use crate::{Column, EpisodeEnd, Error, Handle, Kind, PrioTree, Span, Timeline, Uniform};

/// `expreplay.EpisodeEnd`, a real `enum.Enum` built at import.
static EPISODE_END: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

#[pymodule]
mod expreplay {
    #[pymodule_export]
    use super::{PyColumn, PyPrioTree, PyRollout, PySpan, PyTimeline, PyUniform};

    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = module.py();
        let enums = py.import("enum")?;
        let kwargs = super::PyDict::new(py);
        kwargs.set_item("module", "expreplay")?;

        let episode_end = enums.getattr("Enum")?.call(
            (
                "EpisodeEnd",
                [
                    ("PAD", super::EpisodeEnd::Pad as u8),
                    ("CONTINUE", super::EpisodeEnd::Continue as u8),
                ],
            ),
            Some(&kwargs),
        )?;
        let kind = enums.getattr("IntEnum")?.call(
            (
                "Kind",
                [
                    ("STEP", super::Kind::Step as u8),
                    ("TERMINAL", super::Kind::Terminal as u8),
                    ("TRUNCATED", super::Kind::Truncated as u8),
                    ("FINAL", super::Kind::Final as u8),
                ],
            ),
            Some(&kwargs),
        )?;

        module.add("EpisodeEnd", &episode_end)?;
        module.add("Kind", kind)?;
        super::EPISODE_END.get_or_init(py, || episode_end.unbind());

        Ok(())
    }
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        let msg = value.to_string();
        match value {
            Error::Allocation => PyMemoryError::new_err(msg),
            Error::Env { .. } | Error::Dead(_) | Error::Unwritten(_) => PyIndexError::new_err(msg),
            Error::InvalidArgument(_) | Error::Invalid { .. } | Error::InvalidWeight(_) => {
                PyValueError::new_err(msg)
            }
            // The stored state is degenerate, not the caller's arguments.
            Error::NoWeight | Error::Exhausted => PyRuntimeError::new_err(msg),
        }
    }
}

/// Reads a `[..., 2]` array of handles, returning them flat beside the leading shape.
fn handles_in(handles: &Bound<'_, PyAny>) -> PyResult<(Vec<Handle>, Vec<usize>)> {
    let handles = handles
        .cast::<PyArrayDyn<i64>>()
        .map_err(|_| PyTypeError::new_err("handles must be an int64 array"))?
        .readonly();
    let [lead @ .., 2] = handles.shape() else {
        return Err(PyValueError::new_err(format!(
            "handles must have shape [..., 2], got {:?}",
            handles.shape()
        )));
    };

    let flat = match handles.as_slice() {
        Ok(flat) => Cow::Borrowed(flat),
        Err(_) => Cow::Owned(handles.as_array().iter().copied().collect()),
    };
    if flat.iter().any(|&value| value < 0) {
        return Err(PyValueError::new_err("handles must not be negative"));
    }

    Ok((bytemuck::cast_slice(&flat).to_vec(), lead.to_vec()))
}

type PyHandles<'py> = Bound<'py, PyArrayDyn<i64>>;

fn handles_out<'py>(
    py: Python<'py>,
    handles: Vec<Handle>,
    lead: &[usize],
) -> PyResult<PyHandles<'py>> {
    PyArray1::from_vec(py, bytemuck::cast_vec::<Handle, i64>(handles))
        .reshape([lead, &[2]].concat())
}

/// Where a C-contiguous array's data starts, and its length in bytes. The pointer is dangling
/// for an empty array, and good for as long as `array` is alive otherwise.
fn bytes(array: &Bound<'_, PyUntypedArray>) -> (*mut u8, usize) {
    debug_assert!(array.is_c_contiguous());
    let len = array.len() * array.dtype().itemsize();
    if len == 0 {
        return (std::ptr::NonNull::dangling().as_ptr(), 0);
    }

    // SAFETY: `array` is a live NumPy array, so its object header is readable.
    (unsafe { (*array.as_array_ptr()).data.cast() }, len)
}

/// Which timesteps a draw reads: `length` steps forward from the drawn handle and the
/// `stack - 1` observations before it, with `episode_end` saying what happens at an episode's end.
#[pyclass(name = "Span", module = "expreplay", frozen)]
struct PySpan(Span);

#[pymethods]
impl PySpan {
    #[new]
    #[pyo3(signature = (length, *, stack = 1, episode_end))]
    fn new(length: u32, stack: u32, episode_end: &Bound<'_, PyAny>) -> PyResult<Self> {
        let class = EPISODE_END.get(episode_end.py()).expect("set at import");
        if !episode_end.is_instance(class.bind(episode_end.py()))? {
            return Err(PyTypeError::new_err("episode_end must be an EpisodeEnd"));
        }
        let episode_end = match episode_end.getattr("value")?.extract::<u8>()? {
            0 => EpisodeEnd::Pad,
            1 => EpisodeEnd::Continue,
            value => unreachable!("EpisodeEnd has no member {value}"),
        };

        Ok(Self(Span::new(length, stack, episode_end)?))
    }

    #[getter]
    fn length(&self) -> u32 {
        self.0.length()
    }

    #[getter]
    fn stack(&self) -> u32 {
        self.0.stack()
    }

    #[getter]
    fn episode_end<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let class = EPISODE_END.get(py).expect("set at import");
        class.bind(py).call1((self.0.episode_end() as u8,))
    }

    fn __repr__(&self) -> String {
        format!(
            "Span({}, stack={}, episode_end=EpisodeEnd.{})",
            self.0.length(),
            self.0.stack(),
            match self.0.episode_end() {
                EpisodeEnd::Pad => "PAD",
                EpisodeEnd::Continue => "CONTINUE",
            }
        )
    }
}

/// The timesteps a batch of draws read: `steps`, `frames`, `kinds` and `lengths`.
#[pyclass(name = "Rollout", module = "expreplay", frozen, get_all)]
struct PyRollout {
    steps: Py<PyAny>,
    frames: Py<PyAny>,
    kinds: Py<PyAny>,
    lengths: Py<PyAny>,
}

/// Where each environment's episodes begin and end. Holds no data.
#[pyclass(name = "Timeline", module = "expreplay")]
struct PyTimeline(Timeline);

#[pymethods]
impl PyTimeline {
    #[new]
    fn new(num_envs: usize, env_capacity: usize) -> PyResult<Self> {
        Ok(Self(Timeline::new(num_envs, env_capacity)?))
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.0.num_envs()
    }

    #[getter]
    fn env_capacity(&self) -> usize {
        self.0.env_capacity()
    }

    /// Each environment's newest written timestep, -1 before its first.
    #[getter]
    fn newest<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let newest = (0..self.0.num_envs()).map(|env| self.0.newest(env).map_or(-1, |t| t as i64));
        PyArray1::from_iter(py, newest)
    }

    /// Starts an episode in every environment and returns their opening observations' handles.
    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<PyHandles<'py>> {
        let mut obs = vec![Handle { env: 0, t: 0 }; self.0.num_envs()];
        self.0.reset(&mut obs)?;
        handles_out(py, obs, &[self.0.num_envs()])
    }

    /// Records one environment step each: the incoming observations' handles, and whether each
    /// call completed the transition before it.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        terminated: PyReadonlyArray1<'py, bool>,
        truncated: PyReadonlyArray1<'py, bool>,
    ) -> PyResult<(PyHandles<'py>, Bound<'py, PyArray1<bool>>)> {
        let num_envs = self.0.num_envs();
        let mut obs = vec![Handle { env: 0, t: 0 }; num_envs];
        let mut stepped = vec![false; num_envs];
        (self.0).step(
            terminated.as_slice()?,
            truncated.as_slice()?,
            &mut obs,
            &mut stepped,
        )?;

        Ok((
            handles_out(py, obs, &[num_envs])?,
            PyArray1::from_vec(py, stepped),
        ))
    }

    /// Whether a draw of `span` may start at each handle.
    fn valid<'py>(
        &self,
        py: Python<'py>,
        handles: &Bound<'py, PyAny>,
        span: PyRef<'py, PySpan>,
    ) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
        let (handles, lead) = handles_in(handles)?;
        let valid = handles.iter().map(|&handle| self.0.valid(handle, &span.0));
        PyArray1::from_iter(py, valid).reshape(lead)
    }

    /// The timesteps each draw of `span` reads, in the handles' leading shape.
    fn rollout<'py>(
        &self,
        py: Python<'py>,
        handles: &Bound<'py, PyAny>,
        span: PyRef<'py, PySpan>,
    ) -> PyResult<PyRollout> {
        let (handles, lead) = handles_in(handles)?;
        let rollout = self.0.rollout(&handles, &span.0)?;
        let steps = [&lead[..], &[span.0.length() as usize]].concat();
        let frames = [&lead[..], &[(span.0.stack() + span.0.length()) as usize]].concat();

        Ok(PyRollout {
            steps: handles_out(py, rollout.steps.into_raw_vec_and_offset().0, &steps)?
                .into_any()
                .unbind(),
            frames: handles_out(py, rollout.frames.into_raw_vec_and_offset().0, &frames)?
                .into_any()
                .unbind(),
            kinds: (rollout.kinds.mapv(|kind| kind as u8).into_pyarray(py))
                .reshape(steps)?
                .into_any()
                .unbind(),
            lengths: (rollout.lengths.into_pyarray(py))
                .reshape(lead)?
                .into_any()
                .unbind(),
        })
    }
}

/// One field of every timestep, of any dtype and row shape.
#[pyclass(name = "Column", module = "expreplay")]
struct PyColumn {
    column: Column<u8>,
    dtype: Py<PyArrayDescr>,
    shape: Vec<usize>,
}

#[pymethods]
impl PyColumn {
    #[new]
    #[pyo3(signature = (num_envs, env_capacity, *, dtype, shape = Vec::new()))]
    fn new(
        py: Python<'_>,
        num_envs: usize,
        env_capacity: usize,
        dtype: &Bound<'_, PyAny>,
        shape: Vec<usize>,
    ) -> PyResult<Self> {
        let dtype = PyArrayDescr::new(py, dtype)?;
        if dtype.has_object() {
            return Err(PyTypeError::new_err("a column cannot hold Python objects"));
        }
        let row_bytes = (shape.iter())
            .try_fold(dtype.itemsize(), |bytes, &dim| bytes.checked_mul(dim))
            .ok_or(Error::Allocation)?;
        let column = py.detach(|| Column::new(num_envs, env_capacity, row_bytes))?;

        Ok(Self {
            column,
            dtype: dtype.unbind(),
            shape,
        })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.column.num_envs()
    }

    #[getter]
    fn env_capacity(&self) -> usize {
        self.column.env_capacity()
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        self.dtype.bind(py).clone()
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        pyo3::types::PyTuple::new(py, &self.shape)
    }

    /// Writes one row per handle and returns which landed; a handle already overwritten is
    /// skipped.
    fn write<'py>(
        &mut self,
        py: Python<'py>,
        handles: &Bound<'py, PyAny>,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
        let (handles, lead) = handles_in(handles)?;
        let dtype = self.dtype.bind(py);
        let values = (values.cast::<PyUntypedArray>().ok())
            .filter(|values| values.dtype().is_equiv_to(dtype))
            .ok_or_else(|| {
                PyTypeError::new_err(format!(
                    "values must be a {dtype} array; nothing is cast on the way in"
                ))
            })?;
        let shape = [&lead[..], &self.shape].concat();
        if values.shape() != shape {
            return Err(PyValueError::new_err(format!(
                "values must have shape {shape:?}, got {:?}",
                values.shape()
            )));
        }
        let values = if values.is_c_contiguous() {
            values.clone()
        } else {
            (py.import("numpy")?
                .getattr("ascontiguousarray")?
                .call1((values,))?)
            .cast_into()?
        };

        let mut landed = vec![false; handles.len()];
        let column = &mut self.column;
        let (data, len) = bytes(&values);
        // SAFETY: `values` outlives the slice, which is only read.
        let rows = unsafe { std::slice::from_raw_parts(data, len) };
        py.detach(|| column.write(&handles, rows, Some(&mut landed)))?;

        PyArray1::from_vec(py, landed).reshape(lead)
    }

    /// Reads one row per handle. Raises IndexError on a handle the column does not hold.
    fn read<'py>(
        &self,
        py: Python<'py>,
        handles: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        let (handles, lead) = handles_in(handles)?;
        let shape = [&lead[..], &self.shape].concat();
        let out = (py.import("numpy")?.getattr("empty")?)
            .call1((shape, self.dtype.bind(py)))?
            .cast_into::<PyUntypedArray>()?;

        let (data, len) = bytes(&out);
        // SAFETY: `out` was allocated above and nothing else has seen it.
        let rows = unsafe { std::slice::from_raw_parts_mut(data, len) };
        py.detach(|| self.column.read(&handles, rows))?;

        Ok(out)
    }
}

/// One weight per timestep, drawn in proportion.
#[pyclass(name = "PrioTree", module = "expreplay")]
struct PyPrioTree(PrioTree);

#[pymethods]
impl PyPrioTree {
    #[new]
    #[pyo3(signature = (num_envs, env_capacity, *, seed = None))]
    fn new(
        py: Python<'_>,
        num_envs: usize,
        env_capacity: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        Ok(Self(
            py.detach(|| PrioTree::new(num_envs, env_capacity, seed))?,
        ))
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.0.num_envs()
    }

    #[getter]
    fn env_capacity(&self) -> usize {
        self.0.env_capacity()
    }

    /// The sum of every weight.
    #[getter]
    fn total(&self) -> f32 {
        self.0.total()
    }

    /// Sets one weight per handle and returns which landed; a handle already overwritten is
    /// skipped.
    fn write<'py>(
        &mut self,
        py: Python<'py>,
        handles: &Bound<'py, PyAny>,
        weights: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
        let (handles, lead) = handles_in(handles)?;
        let weights = weights
            .cast::<PyArrayDyn<f32>>()
            .map_err(|_| PyTypeError::new_err("weights must be a float32 array"))?
            .readonly();
        if weights.shape() != lead {
            return Err(PyValueError::new_err(format!(
                "weights must have shape {lead:?}, got {:?}",
                weights.shape()
            )));
        }
        let weights: Vec<f32> = weights.as_array().iter().copied().collect();

        let mut landed = vec![false; handles.len()];
        self.0.write(&handles, &weights, Some(&mut landed))?;

        PyArray1::from_vec(py, landed).reshape(lead)
    }

    /// Reads one weight per handle. Raises IndexError on a handle the tree does not hold.
    fn read<'py>(
        &self,
        py: Python<'py>,
        handles: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let (handles, lead) = handles_in(handles)?;
        PyArray1::from_vec(py, self.0.read(&handles)?).reshape(lead)
    }

    /// Draws `batch` handles in proportion to weight among those `span` can start at on
    /// `timeline`, and returns each one's weight beside it.
    #[pyo3(signature = (batch, timeline, span, *, stratified = false))]
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        batch: usize,
        timeline: PyRef<'py, PyTimeline>,
        span: PyRef<'py, PySpan>,
        stratified: bool,
    ) -> PyResult<(PyHandles<'py>, Bound<'py, PyArray1<f32>>)> {
        let (tree, timeline, span) = (&mut self.0, &timeline.0, &span.0);
        let (handles, weights) = py.detach(|| tree.sample(batch, timeline, span, stratified))?;

        Ok((
            handles_out(py, handles, &[batch])?,
            PyArray1::from_vec(py, weights),
        ))
    }
}

/// Draws uniformly over everything a timeline holds.
#[pyclass(name = "Uniform", module = "expreplay")]
struct PyUniform(Uniform);

#[pymethods]
impl PyUniform {
    #[new]
    #[pyo3(signature = (*, seed = None))]
    fn new(seed: Option<u64>) -> Self {
        Self(Uniform::new(seed))
    }

    /// Draws `batch` handles uniformly among those `span` can start at on `timeline`.
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        batch: usize,
        timeline: PyRef<'py, PyTimeline>,
        span: PyRef<'py, PySpan>,
    ) -> PyResult<PyHandles<'py>> {
        let (uniform, timeline, span) = (&mut self.0, &timeline.0, &span.0);
        let handles = py.detach(|| uniform.sample(batch, timeline, span))?;

        handles_out(py, handles, &[batch])
    }
}
