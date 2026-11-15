//! [`Column`]: one field of every timestep.

use bytemuck::Pod;
use rayon::prelude::*;

use crate::ring::Ring;
use crate::{Error, Handle, try_zeroed_vec};

/// Bytes past which a read is handed to rayon. Below it the fork-join costs more than the copy.
const PAR_READ_MIN_BYTES: usize = 1 << 22;

/// A field stored per timestep, `row_len` elements each. It only copies, so an element type
/// known at runtime is a `Column<u8>` with the row length in bytes.
pub struct Column<T> {
    ring: Ring,
    row_len: usize,
    data: Box<[T]>,
}

impl<T: Pod + Send + Sync> Column<T> {
    pub fn new(num_envs: usize, env_capacity: usize, row_len: usize) -> Result<Self, Error> {
        if row_len == 0 {
            return Err(Error::InvalidArgument(
                "a column's rows must hold at least one element".to_owned(),
            ));
        }
        let ring = Ring::new(num_envs, env_capacity)?;
        let len = ring.len().checked_mul(row_len).ok_or(Error::Allocation)?;

        Ok(Self {
            ring,
            row_len,
            data: try_zeroed_vec(len)?.into_boxed_slice(),
        })
    }

    pub fn num_envs(&self) -> usize {
        self.ring.num_envs()
    }

    pub fn env_capacity(&self) -> usize {
        self.ring.env_capacity()
    }

    pub fn row_len(&self) -> usize {
        self.row_len
    }

    /// Writes row `i` of `rows` at `handles[i]`, in order, and returns how many landed.
    ///
    /// A handle the ring has moved past is skipped, as `landed` reports. One beyond the newest
    /// timestep advances the ring, zeroing the timesteps in between.
    pub fn write(
        &mut self,
        handles: &[Handle],
        rows: &[T],
        mut landed: Option<&mut [bool]>,
    ) -> Result<usize, Error> {
        check_len("rows", rows.len(), handles.len() * self.row_len)?;
        if let Some(landed) = &landed {
            check_len("landed", landed.len(), handles.len())?;
        }

        let mut count = 0;
        for (i, (&handle, src)) in handles
            .iter()
            .zip(rows.chunks_exact(self.row_len))
            .enumerate()
        {
            let claim = self.ring.claim(handle)?;
            if let Some(landed) = &mut landed {
                landed[i] = claim.is_some();
            }
            let Some((row, skipped)) = claim else {
                continue;
            };

            for skipped in skipped {
                self.data[skipped * self.row_len..][..self.row_len].fill(T::zeroed());
            }
            self.data[row * self.row_len..][..self.row_len].copy_from_slice(src);
            count += 1;
        }

        Ok(count)
    }

    /// Copies the row at `handles[i]` into row `i` of `out`.
    ///
    /// Fails with [`Error::Dead`] or [`Error::Unwritten`] on a handle the column does not hold,
    /// leaving `out` unspecified.
    pub fn read(&self, handles: &[Handle], out: &mut [T]) -> Result<(), Error> {
        check_len("out", out.len(), handles.len() * self.row_len)?;

        let copy = |(dst, &handle): (&mut [T], &Handle)| {
            let row = self.ring.locate(handle)?;
            dst.copy_from_slice(&self.data[row * self.row_len..][..self.row_len]);
            Ok(())
        };

        if size_of_val(out) < PAR_READ_MIN_BYTES {
            out.chunks_exact_mut(self.row_len)
                .zip(handles)
                .try_for_each(copy)
        } else {
            (out.par_chunks_exact_mut(self.row_len).zip(handles)).try_for_each(copy)
        }
    }
}

pub(crate) fn check_len(name: &str, len: usize, expected: usize) -> Result<(), Error> {
    if len != expected {
        return Err(Error::InvalidArgument(format!(
            "{name} must hold {expected} elements, got {len}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle(env: u64, t: u64) -> Handle {
        Handle { env, t }
    }

    #[test]
    fn test_column_reads_back_what_was_written() {
        let mut column = Column::<u16>::new(2, 4, 2).unwrap();
        let handles = [handle(0, 0), handle(1, 0), handle(0, 1)];
        assert_eq!(
            column.write(&handles, &[1, 2, 3, 4, 5, 6], None).unwrap(),
            3
        );

        let mut out = [0; 4];
        column
            .read(&[handle(0, 1), handle(1, 0)], &mut out)
            .unwrap();
        assert_eq!(out, [5, 6, 3, 4]);

        // The same handle may be read twice, and written twice: the later write wins.
        column
            .write(&[handle(0, 1), handle(0, 1)], &[7, 7, 8, 8], None)
            .unwrap();
        column
            .read(&[handle(0, 1), handle(0, 1)], &mut out)
            .unwrap();
        assert_eq!(out, [8, 8, 8, 8]);
    }

    #[test]
    fn test_column_refuses_handles_it_does_not_hold() {
        let mut column = Column::<u8>::new(1, 4, 1).unwrap();
        let mut out = [0];

        assert!(matches!(
            column.read(&[handle(0, 0)], &mut out),
            Err(Error::Unwritten(_))
        ));
        assert!(matches!(
            column.read(&[handle(1, 0)], &mut out),
            Err(Error::Env { .. })
        ));

        for t in 0..6 {
            column.write(&[handle(0, t)], &[t as u8], None).unwrap();
        }
        // Timesteps 2..6 are the four still in the ring.
        assert!(matches!(
            column.read(&[handle(0, 1)], &mut out),
            Err(Error::Dead(_))
        ));
        column.read(&[handle(0, 2)], &mut out).unwrap();
        assert_eq!(out, [2]);
        assert!(matches!(
            column.read(&[handle(0, 6)], &mut out),
            Err(Error::Unwritten(_))
        ));
    }

    #[test]
    fn test_column_skips_stale_writes_and_says_so() {
        let mut column = Column::<u8>::new(1, 4, 1).unwrap();
        for t in 0..6 {
            column.write(&[handle(0, t)], &[t as u8], None).unwrap();
        }

        let mut landed = [true; 3];
        let handles = [handle(0, 1), handle(0, 5), handle(0, 2)];
        let count = column
            .write(&handles, &[91, 95, 92], Some(&mut landed))
            .unwrap();
        assert_eq!((count, landed), (2, [false, true, true]));

        // The stale write must not have landed on timestep 5, which shares its row.
        let mut out = [0; 4];
        let live = [handle(0, 2), handle(0, 3), handle(0, 4), handle(0, 5)];
        column.read(&live, &mut out).unwrap();
        assert_eq!(out, [92, 3, 4, 95]);
    }

    #[test]
    fn test_column_zeroes_the_timesteps_a_write_skips() {
        let mut column = Column::<u8>::new(1, 4, 1).unwrap();
        for t in 0..4 {
            column
                .write(&[handle(0, t)], &[10 + t as u8], None)
                .unwrap();
        }

        // Skipping 4 and 5 leaves their rows holding timesteps 0 and 1 of the lap before.
        column.write(&[handle(0, 6)], &[16], None).unwrap();
        let mut out = [0; 4];
        let live = [handle(0, 3), handle(0, 4), handle(0, 5), handle(0, 6)];
        column.read(&live, &mut out).unwrap();
        assert_eq!(out, [13, 0, 0, 16]);

        // A jump of more than a lap clears everything but the row written.
        column.write(&[handle(0, 100)], &[1], None).unwrap();
        let live = [handle(0, 97), handle(0, 98), handle(0, 99), handle(0, 100)];
        column.read(&live, &mut out).unwrap();
        assert_eq!(out, [0, 0, 0, 1]);
    }

    #[test]
    fn test_column_reads_agree_across_the_parallel_threshold() {
        let row_len = PAR_READ_MIN_BYTES / 8;
        let mut column = Column::<u32>::new(2, 3, row_len).unwrap();
        let handles = [handle(0, 0), handle(1, 0), handle(0, 1), handle(1, 1)];
        let rows: Vec<u32> = (0..4)
            .flat_map(|i| std::iter::repeat_n(i, row_len))
            .collect();
        column.write(&handles, &rows, None).unwrap();

        let order = [handle(1, 1), handle(0, 0), handle(0, 1), handle(1, 0)];
        let mut out = vec![0; 4 * row_len];
        column.read(&order, &mut out).unwrap();
        for (row, expected) in out.chunks_exact(row_len).zip([3, 0, 2, 1]) {
            assert!(row.iter().all(|&value| value == expected));
        }

        assert!(matches!(
            column.read(
                &[handle(0, 0), handle(0, 2), handle(0, 0), handle(0, 0)],
                &mut out
            ),
            Err(Error::Unwritten(_))
        ));
    }
}
