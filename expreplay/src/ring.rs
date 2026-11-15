//! The bookkeeping [`Column`](crate::Column) and [`PrioTree`](crate::PrioTree) share: one ring of
//! `env_capacity` rows per environment, addressed by [`Handle`].

use std::ops::Range;

use crate::{Error, Handle};

pub(crate) struct Ring {
    env_capacity: u64,
    /// Per environment: one past the newest timestep written. Row `t % env_capacity` holds
    /// timestep `t` for the `env_capacity` timesteps below it, and those are the live ones.
    written: Box<[u64]>,
}

impl Ring {
    pub(crate) fn new(num_envs: usize, env_capacity: usize) -> Result<Self, Error> {
        if num_envs == 0 || env_capacity == 0 {
            return Err(Error::InvalidArgument(
                "num_envs and env_capacity must both be at least one".to_owned(),
            ));
        }
        num_envs
            .checked_mul(env_capacity)
            .ok_or(Error::Allocation)?;

        Ok(Self {
            env_capacity: env_capacity as u64,
            written: vec![0; num_envs].into_boxed_slice(),
        })
    }

    pub(crate) fn num_envs(&self) -> usize {
        self.written.len()
    }

    pub(crate) fn env_capacity(&self) -> usize {
        self.env_capacity as usize
    }

    /// Rows in the whole ring.
    pub(crate) fn len(&self) -> usize {
        self.num_envs() * self.env_capacity()
    }

    fn row(&self, env: usize, t: u64) -> usize {
        env * self.env_capacity() + (t % self.env_capacity) as usize
    }

    fn env(&self, handle: Handle) -> Result<usize, Error> {
        let num_envs = self.num_envs();
        if num_envs as u64 <= handle.env {
            return Err(Error::Env {
                env: handle.env,
                num_envs,
            });
        }

        Ok(handle.env as usize)
    }

    /// The row holding `handle`, for a read.
    pub(crate) fn locate(&self, handle: Handle) -> Result<usize, Error> {
        let env = self.env(handle)?;
        let written = self.written[env];

        if written <= handle.t {
            return Err(Error::Unwritten(handle));
        }
        if self.env_capacity < written - handle.t {
            return Err(Error::Dead(handle));
        }

        Ok(self.row(env, handle.t))
    }

    /// The row a write to `handle` goes to, or `None` where the ring has moved past it. A write
    /// beyond the newest timestep advances the ring and also returns the rows it skipped, which
    /// still hold the previous lap and are the caller's to zero.
    pub(crate) fn claim(
        &mut self,
        handle: Handle,
    ) -> Result<Option<(usize, impl Iterator<Item = usize>)>, Error> {
        let env = self.env(handle)?;
        let written = self.written[env];

        if written.saturating_sub(self.env_capacity) > handle.t {
            return Ok(None);
        }

        let skipped: Range<u64> = if written <= handle.t {
            self.written[env] = handle.t + 1;
            written.max((handle.t + 1).saturating_sub(self.env_capacity))..handle.t
        } else {
            0..0
        };

        let this = &*self;
        Ok(Some((
            this.row(env, handle.t),
            skipped.map(move |t| this.row(env, t)),
        )))
    }

    /// The live timestep a row holds, or `None` where nothing has been written to it.
    pub(crate) fn handle_at(&self, row: usize) -> Option<Handle> {
        let (env, slot) = (
            row / self.env_capacity(),
            (row % self.env_capacity()) as u64,
        );
        let newest = self.written[env].checked_sub(1)?;
        let back = (newest % self.env_capacity + self.env_capacity - slot) % self.env_capacity;

        Some(Handle {
            env: env as u64,
            t: newest.checked_sub(back)?,
        })
    }
}
