//! Experience replay for vectorised environments, as independent parts that agree on one thing:
//! a [`Handle`] names an observation by its environment and its timestep there.
//!
//! - [`Timeline`] knows where episodes begin and end. It holds no data.
//! - [`Column`] holds one field per timestep. It knows nothing of episodes.
//! - [`PrioTree`] holds one weight per timestep and draws in proportion to them.
//! - [`Uniform`] draws over everything a timeline has written.
//!
//! A sampler takes the timeline it draws against and keeps only what [`Timeline::valid`] accepts.
//! Nothing here computes on a stored value: returns, discounts and priorities are the caller's.

mod column;
mod prio_tree;
mod pyffi;
mod ring;
mod sampling;
mod sum_tree;
mod timeline;
mod uniform;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

pub use column::Column;
pub use prio_tree::PrioTree;
pub use sampling::{MAX_ATTEMPTS, STRATIFIED_ATTEMPTS};
pub use timeline::{Rollout, Timeline};
pub use uniform::Uniform;

/// One observation: the `t`-th its environment wrote. The action taken from it and the reward
/// that earned share the handle.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct Handle {
    pub env: u64,
    pub t: u64,
}

/// What a rollout does on reaching the end of its episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EpisodeEnd {
    /// Stay inside the episode: the steps past a terminal are padding, and a truncation is
    /// accepted only as the last step.
    Pad = 0,
    /// Run on into the following episodes, with [`Rollout::kinds`] marking each boundary.
    Continue = 1,
}

/// The stretch of timesteps a draw reads: `length` steps forward from the drawn handle, and the
/// `stack - 1` observations before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    length: u32,
    stack: u32,
    episode_end: EpisodeEnd,
}

impl Span {
    pub fn new(length: u32, stack: u32, episode_end: EpisodeEnd) -> Result<Self, Error> {
        if length == 0 || stack == 0 {
            return Err(Error::InvalidArgument(
                "a span's length and stack must both be at least one".to_owned(),
            ));
        }

        Ok(Self {
            length,
            stack,
            episode_end,
        })
    }

    pub fn length(&self) -> u32 {
        self.length
    }

    pub fn stack(&self) -> u32 {
        self.stack
    }

    pub fn episode_end(&self) -> EpisodeEnd {
        self.episode_end
    }
}

/// What a timestep holds beyond its observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    /// A transition whose episode carries on into the next timestep.
    Step = 0,
    /// A transition that ended its episode: nothing is bootstrapped past it.
    Terminal = 1,
    /// A transition cut short by a time limit or a mid-run reset. The next timestep holds the
    /// observation to bootstrap from.
    Truncated = 2,
    /// An observation with no action: the one an episode ended on.
    Final = 3,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("allocation failed")]
    Allocation,
    #[error("{0}")]
    InvalidArgument(String),
    #[error("environment {env} is out of range for {num_envs} environments")]
    Env { env: u64, num_envs: usize },
    #[error("timestep {} of environment {} has been overwritten", .0.t, .0.env)]
    Dead(Handle),
    #[error("timestep {} of environment {} has not been written", .0.t, .0.env)]
    Unwritten(Handle),
    #[error("timestep {} of environment {} cannot start a rollout of {span:?}", .handle.t, .handle.env)]
    Invalid { handle: Handle, span: Span },
    #[error("weight {0} is not finite and non-negative")]
    InvalidWeight(f32),
    #[error("every weight is zero, so there is nothing to draw")]
    NoWeight,
    #[error("gave up finding a valid timestep after {MAX_ATTEMPTS} attempts")]
    Exhausted,
}

/// `vec![0; len]` that fails rather than aborts: a column can be tens of gigabytes.
pub(crate) fn try_zeroed_vec<T: Zeroable>(len: usize) -> Result<Vec<T>, Error> {
    bytemuck::allocation::try_zeroed_vec(len).map_err(|()| Error::Allocation)
}
