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
pub fn try_zeroed_vec<T: Zeroable>(len: usize) -> Result<Vec<T>, AllocationError> {
    allocation::try_zeroed_vec(len).map_err(|()| AllocationError::new())
}

/// Number of elements past which a per-row copy is handed to rayon.
///
/// A batch of stacked Atari observations is a few hundred kilobytes and is worth spreading over
/// the pool; a batch of one reward or one action per environment is a few hundred *bytes*, where
/// a fork-join costs far more than the copy itself. The exact cut-off does not matter much --
/// anything near it is cheap whichever way it goes.
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
/// indices, one per output row. A stacked output is gathered by passing `batch * stack` slots,
/// oldest frame of each stack first, and reshaping the result afterwards: because the stack axis
/// leads, every frame is a contiguous run of bytes and each copy is a memcpy rather than a
/// stride-`stack` scatter.
///
/// The destination is allocated uninitialised and written through [`MaybeUninit`], rather than
/// zeroed and then overwritten. A batch of stacked observations is megabytes and every byte of it
/// is replaced here, so the zeroing would buy nothing. Note that this is the one array in the
/// crate not allocated through [`try_zeroed_vec`]: it is a per-batch temporary a few megabytes
/// wide rather than the multi-gigabyte storage that rule is about, and `uninit` has no fallible
/// form to reach for.
///
/// [`MaybeUninit`]: std::mem::MaybeUninit
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

    // A batch of stacked observations is worth spreading over the pool; a batch of one action
    // apiece is not. See [`PAR_COPY_MIN_ELEMS`].
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

    // SAFETY: `out` was shaped with exactly `slots.len()` rows, and `azip!` zips it against
    // `slots` -- a length mismatch would have panicked rather than skipped a row -- so the loop
    // above visited every row, and `assign_to` writes every element of the row it is handed.
    // Nothing is left uninitialised.
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
