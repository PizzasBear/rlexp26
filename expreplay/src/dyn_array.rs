//! [`DynArray`], an owned or borrowed `ndarray` whose element type is only known at runtime.
//!
//! The buffer stores observations and actions in whatever dtype the caller asked for, so every
//! array it touches is one of these. [`for_all_dtypes`] is the single list of supported types:
//! each declaration below is generated from it, so adding a dtype means editing that one macro.

use std::marker::PhantomData;

use ndarray::prelude::*;
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyArrayDyn};
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::utils::{AllocationError, gather_batch, try_zeroed_vec, write_batch};

/// Applies `$macro` to the full list of supported dtypes, as `Variant => rust_type` pairs.
///
/// Every dtype-dispatching declaration in the crate is generated from this one list, so a new
/// element type is added here and nowhere else.
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
            F32 => f32,
            F64 => f64,
        }
    };
}

macro_rules! def_dtype {
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum DType {
            $($variant,)+
        }

        impl DType {
            /// The NumPy descriptor for this dtype.
            pub fn np<'py>(self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
                match self {
                    $(Self::$variant => numpy::dtype::<$ty>(py),)+
                }
            }

            /// Matches a NumPy dtype, or anything NumPy can read as one, against the supported
            /// list. Rejection names every dtype that would have been accepted, built from the
            /// same list rather than written out by hand.
            pub fn from_np<'py>(dtype: &Bound<'py, PyAny>) -> PyResult<Self> {
                let py = dtype.py();

                let descr = PyArrayDescr::new(py, dtype)?;

                $(if descr.is_equiv_to(&numpy::dtype::<$ty>(py)) {
                    Ok(Self::$variant)
                } else)+ {
                    Err(PyTypeError::new_err(format!(
                        "Unsupported dtype {descr}, ReplayBuffer supports {}",
                        [$(numpy::dtype::<$ty>(py).to_string(),)+].join(", "),
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
            /// Allocates a zeroed array of `shape`, failing rather than aborting if it does not
            /// fit.
            ///
            /// This is by far the largest allocation the buffer makes -- for Atari-sized
            /// observations it is three orders of magnitude bigger than everything else here put
            /// together -- so it is the one that most needs to surface as a `MemoryError` instead
            /// of taking the interpreter with it. See [`try_zeroed_vec`].
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

        /// [`write_batch`] over a pair of arrays whose dtype is only known at runtime.
        ///
        /// `dst` stays generic over its representation, as `src` is: the `DataMut` bound per dtype
        /// is what says "anything writable", so a borrowed repr is accepted the day one exists
        /// without this signature having to change to meet it.
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
                // Both sides come from the same stored array's dtype -- `src` is extracted with
                // it at the boundary and `dst` is the array it was extracted for -- so a mismatch
                // is a bug here rather than anything the caller can provoke.
                (dst, src) => unreachable!(
                    "cannot write a {:?} batch into a {:?} array",
                    src.dtype(),
                    dst.dtype(),
                ),
            }
        }

        /// Gathers `slots` out of a stored `[env, slot, ..tail]` array into a fresh
        /// `[slots.len(), ..tail]` one of the same dtype.
        ///
        /// The gather sees one row per slot, so a stacked observation comes back with its stack
        /// folded into the batch axis: `obs_stack` consecutive rows per draw, for the caller to
        /// reshape. That reshape is free, since the array is allocated here and is contiguous.
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
