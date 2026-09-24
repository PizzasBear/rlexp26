//! Generic array and allocation helpers, with nothing in them specific to a replay buffer.

use bytemuck::{Zeroable, allocation};
use ndarray::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("Allocation failed")]
pub struct AllocationError(());

impl AllocationError {
    pub(crate) const fn new() -> Self {
        Self(())
    }
}

/// Allocates `len` zeroed `T`s, returning an error where `vec![0; n]` or `Array::zeros` would
/// abort through [`handle_alloc_error`]. The observation array is routinely tens of gigabytes,
/// and asking for too much should reach Python as a `MemoryError`.
///
/// Goes through `alloc_zeroed`, so fresh pages are mapped lazily rather than memset up front.
/// Under Linux's default overcommit a too-large request usually succeeds here and fails later,
/// when the pages are touched.
///
/// [`handle_alloc_error`]: std::alloc::handle_alloc_error
pub fn try_zeroed_vec<T: Zeroable>(len: usize) -> Result<Vec<T>, AllocationError> {
    allocation::try_zeroed_vec(len).map_err(|()| AllocationError::new())
}

/// Elements past which a copy is handed to rayon: a batch of stacked frames is worth a fork-join,
/// a batch of one reward per environment is not.
pub const PAR_COPY_MIN_ELEMS: usize = 1 << 14;

/// Copies row `env` of `src` into `dst[env, slots[env]]`, for every environment at once.
///
/// `dst` is a stored `[env, slot, ..]` array and `src` the per-environment batch that goes into it.
pub fn write_batch<T, D>(dst: &mut ArrayRef<T, D>, slots: &[u32], src: &ArrayRef<T, D::Smaller>)
where
    T: Copy + Send + Sync + 'static,
    D: Dimension + ndarray::RemoveAxis,
    D::Smaller: ndarray::RemoveAxis,
{
    if src.len() < PAR_COPY_MIN_ELEMS {
        ndarray::azip!((mut dst in dst.outer_iter_mut(), &slot in slots, src in src.outer_iter()) {
            dst.index_axis_mut(Axis(0), slot as _).assign(&src);
        });
    } else {
        ndarray::par_azip!((mut dst in dst.outer_iter_mut(), &slot in slots, src in src.outer_iter()) {
            dst.index_axis_mut(Axis(0), slot as _).assign(&src);
        });
    }
}

/// Gathers one stored slot per row into a fresh `[slots.len(), ..tail]` array.
///
/// `src` is a stored `[env, slot, ..tail]` array and `slots` holds flat `env * capacity + slot`
/// indices. A stack is gathered as `batch * stack` slots, oldest frame first, for the caller to
/// reshape; the leading stack axis makes each frame one memcpy.
///
/// The output is allocated uninitialised, since every byte is overwritten. It is a per-batch
/// temporary, so the fallible-allocation rule for the storage does not apply.
pub fn gather_batch<T, D>(
    src: &ArrayRef<T, D>,
    slots: &[u32],
    capacity: usize,
) -> Array<T, D::Smaller>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension + ndarray::RemoveAxis,
    D::Smaller: ndarray::RemoveAxis,
{
    // `src` is `[env, slot, ..tail]`, so everything past the slot axis is one gathered row.
    let mut shape = D::Smaller::zeros(src.ndim() - 1);
    shape[0] = slots.len();
    shape.slice_mut()[1..].copy_from_slice(&src.shape()[2..]);

    let mut out = Array::<T, D::Smaller>::uninit(shape);

    if out.len() < PAR_COPY_MIN_ELEMS {
        ndarray::azip!((mut dst in out.outer_iter_mut(), &flat in slots) {
            let (env, slot) = (flat as usize / capacity, flat as usize % capacity);
            (src.index_axis(Axis(0), env).index_axis(Axis(0), slot)).assign_to(&mut dst);
        });
    } else {
        ndarray::par_azip!((mut dst in out.outer_iter_mut(), &flat in slots) {
            let (env, slot) = (flat as usize / capacity, flat as usize % capacity);
            (src.index_axis(Axis(0), env).index_axis(Axis(0), slot)).assign_to(&mut dst);
        });
    }

    // SAFETY: `out` has exactly `slots.len()` rows, `azip!` panics on a length mismatch rather than
    // skipping one, and `assign_to` writes every element of each row.
    unsafe { out.assume_init() }
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
    fn test_gather_pulls_slots_out_of_the_stored_layout() {
        // Stored as `[env=2, slot=3, 2]`, with the value encoding env/slot/element.
        let src = ArrayD::from_shape_fn(IxDyn(&[2, 3, 2]), |i| {
            (100 * i[0] + 10 * i[1] + i[2]) as u32
        });

        // Two frames per batch element, gathered as `[batch * stack, 2]` for the caller to
        // reshape. Flat slots: env 0 slots 0 and 1, then env 1 slots 1 and 2.
        let dst = gather_batch(&src, &[0, 1, 4, 5], 3);
        let expected = vec![0, 1, 10, 11, 110, 111, 120, 121];
        assert_eq!(
            dst.into_shape_with_order(IxDyn(&[2, 2, 2])).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), expected).unwrap()
        );

        // No stack: one row per slot, and the same function serves.
        let dst = gather_batch(&src, &[4, 0], 3);
        assert_eq!(
            dst,
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![110, 111, 0, 1]).unwrap()
        );
    }

    #[test]
    fn test_gather_fills_every_row_across_the_parallel_threshold() {
        // Every element of the result is written by the gather, which is what the `assume_init`
        // in it rests on. Sized past the rayon cut-off so the parallel branch is the one checked.
        let cols = PAR_COPY_MIN_ELEMS;
        let src = ArrayD::from_shape_fn(IxDyn(&[2, 3, cols]), |i| (100 * i[0] + 10 * i[1]) as u32);

        let dst = gather_batch(&src, &[5, 1, 4, 0], 3);

        assert_eq!(dst.shape(), [4, cols]);
        for (row, &flat) in [5u32, 1, 4, 0].iter().enumerate() {
            let (env, slot) = (flat as usize / 3, flat as usize % 3);
            assert!(
                (dst.index_axis(Axis(0), row).iter()).all(|&v| v == (100 * env + 10 * slot) as u32)
            );
        }
    }
}
