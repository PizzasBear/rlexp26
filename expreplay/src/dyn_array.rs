//! [`DynArray`], an owned or borrowed `ndarray` whose element type is only known at runtime.

use std::marker::PhantomData;

use ndarray::prelude::*;
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyArrayDyn};
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::utils::{AllocationError, gather_batch, try_zeroed_vec, write_batch};

/// Applies `$macro` to the supported dtypes, as `Variant => rust_type` pairs. Every
/// dtype-dispatching declaration in the crate is generated from this list.
macro_rules! for_all_dtypes {
    ($macro:ident) => {
        $macro! {
            U8  => u8,
            U16 => u16,
            U32 => u32,
            U64 => u64,
            I8  => i8,
            I16 => i16,
            I32 => i32,
            I64 => i64,
            F16 => half::f16,
            BF16 => half::bf16,
            F32 => f32,
            F64 => f64,
        }
    };
}

/// The NumPy descriptor for a supported dtype, or `None` where this interpreter has none. Only
/// `bfloat16` can be missing: `ml_dtypes` registers it, and rust-numpy's `Element` impl panics
/// without it.
macro_rules! np_descr {
    (BF16, $py:expr, $ty:ty) => {
        PyArrayDescr::new($py, "bfloat16").ok()
    };
    ($variant:ident, $py:expr, $ty:ty) => {
        Some(numpy::dtype::<$ty>($py))
    };
}

macro_rules! def_dtype {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum DType {
            $($variant,)+
        }

        impl DType {
            /// The NumPy descriptor for this dtype. Infallible: a `DType` names only a dtype
            /// [`Self::from_np`] found.
            pub fn np<'py>(self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
                match self {
                    $(Self::$variant => numpy::dtype::<$ty>(py),)+
                }
            }

            /// Matches a NumPy dtype, or anything NumPy reads as one, against the supported list.
            /// The rejection lists the supported dtypes this interpreter has.
            pub fn from_np<'py>(dtype: &Bound<'py, PyAny>) -> PyResult<Self> {
                let py = dtype.py();

                let descr = PyArrayDescr::new(py, dtype)?;

                $(if np_descr!($variant, py, $ty).is_some_and(|d| descr.is_equiv_to(&d)) {
                    Ok(Self::$variant)
                } else)+ {
                    Err(PyTypeError::new_err(format!(
                        "Unsupported dtype {descr}, ReplayBuffer supports {}",
                        [$(np_descr!($variant, py, $ty),)+]
                            .into_iter()
                            .flatten()
                            .map(|d| d.to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                    )))
                }
            }
        }
    };
}
pub(crate) use for_all_dtypes;

for_all_dtypes!(def_dtype);

macro_rules! def_dyn_array_repr_trait {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        pub trait DynArrayRepr {
            $(type $variant: ndarray::Data<Elem = $ty>;)+
        }
    };
}
for_all_dtypes!(def_dyn_array_repr_trait);

macro_rules! def_dyn_array_owned_repr_struct {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        pub struct DynArrayOwnedRepr(());
        impl DynArrayRepr for DynArrayOwnedRepr {
            $(type $variant = ndarray::OwnedRepr<$ty>;)+
        }
    };
}
for_all_dtypes!(def_dyn_array_owned_repr_struct);

macro_rules! def_dyn_array_view_repr_struct {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        pub struct DynArrayViewRepr<'a>(PhantomData<&'a ()>);
        impl<'a> DynArrayRepr for DynArrayViewRepr<'a> {
            $(type $variant = ndarray::ViewRepr<&'a $ty>;)+
        }
    };
}
for_all_dtypes!(def_dyn_array_view_repr_struct);

macro_rules! def_dyn_array_base {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        #[derive(PartialEq, Debug)]
        pub enum DynArrayBase<A: DynArrayRepr> {
            $($variant(ArrayBase<A::$variant, IxDyn>),)+
        }
    };
}

for_all_dtypes!(def_dyn_array_base);

pub type DynArray = DynArrayBase<DynArrayOwnedRepr>;
pub type DynArrayView<'a> = DynArrayBase<DynArrayViewRepr<'a>>;

macro_rules! impl_dyn_array_methods {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        impl<A: DynArrayRepr> DynArrayBase<A> {
            #[inline]
            pub const fn dtype(&self) -> DType {
                match self {
                    $(Self::$variant(_) => DType::$variant,)+
                }
            }

            #[inline]
            pub fn ndim(&self) -> usize {
                match self {
                    $(Self::$variant(array) => array.ndim(),)+
                }
            }

            #[inline]
            pub fn shape(&self) -> &[usize] {
                match self {
                    $(Self::$variant(array) => array.shape(),)+
                }
            }

            pub fn into_shape_with_order<S>(self, shape: S) -> Result<Self, ndarray::ShapeError>
            where
                S: ndarray::ShapeArg<Dim = IxDyn>,
            {
                match self {
                    $(Self::$variant(array) => {
                        Ok(Self::$variant(array.into_shape_with_order(shape)?))
                    })+
                }
            }
        }

        impl DynArray {
            /// Allocates a zeroed array, failing instead of aborting; see [`try_zeroed_vec`].
            pub fn try_zeros(shape: &[usize], dtype: DType) -> Result<Self, AllocationError> {
                let len = (shape.iter())
                    .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                    .ok_or_else(AllocationError::new)?;

                match dtype {
                    $(DType::$variant => {
                        let array = ArrayD::from_shape_vec(IxDyn(shape), try_zeroed_vec(len)?)
                            .expect("`len` is the product of the shape, so the array cannot reject it");
                        Ok(Self::$variant(array))
                    })+
                }
            }

            pub fn into_np<'py>(self, py: Python<'py>) -> Bound<'py, PyAny> {
                match self {
                    $(Self::$variant(array) => {
                        PyArrayDyn::from_owned_array(py, array).into_any()
                    })+
                }
            }
        }

        impl<A: DynArrayRepr> DynArrayBase<A> {
            /// Narrows to one index along `axis`, dropping that axis from the shape.
            pub fn index_axis_move(self, axis: Axis, index: usize) -> Self {
                match self {
                    $(Self::$variant(array) => {
                        Self::$variant(array.index_axis_move(axis, index))
                    })+
                }
            }

        }

        /// [`write_batch`] over arrays whose dtype is only known at runtime.
        pub fn dyn_write_batch<DA, SA>(
            dst: &mut DynArrayBase<DA>,
            slots: &[u32],
            src: &DynArrayBase<SA>,
        )
        where
            SA: DynArrayRepr,
            DA: DynArrayRepr,
            $(DA::$variant: ndarray::DataMut,)+
        {
            match (dst, src) {
                $((DynArrayBase::$variant(dst), DynArrayBase::$variant(src)) => {
                    write_batch(dst, slots, src)
                })+
                // `src` was extracted as `dst`'s dtype.
                (dst, src) => unreachable!(
                    "cannot write a {:?} batch into a {:?} array",
                    src.dtype(),
                    dst.dtype(),
                ),
            }
        }

        /// Gathers `slots` out of a stored `[env, slot, ..tail]` array into a fresh
        /// `[slots.len(), ..tail]` one; see [`gather_batch`].
        pub fn dyn_gather<A: DynArrayRepr>(
            src: &DynArrayBase<A>,
            slots: &[u32],
            capacity: usize,
        ) -> DynArray {
            match src {
                $(DynArrayBase::$variant(src) => {
                    DynArray::$variant(gather_batch(src, slots, capacity))
                })+
            }
        }
    };
}
for_all_dtypes!(impl_dyn_array_methods);
