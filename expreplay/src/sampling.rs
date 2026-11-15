//! The accept loop every memoryless sampler shares.

use crate::{Error, Handle};

/// Candidates per drawn handle before giving up.
pub const MAX_ATTEMPTS: u32 = 128;

/// Attempts a stratified draw spends in its own stratum before widening to the whole mass: a
/// stratum can sit wholly inside the invalid run behind a write head.
pub const STRATIFIED_ATTEMPTS: u32 = 8;

/// Draws `batch` handles, each the first of its `candidate(k, attempt)`s that `valid` accepts.
/// Exact so long as a candidate does not depend on what was rejected before it.
pub(crate) fn fill(
    batch: usize,
    mut candidate: impl FnMut(usize, u32) -> Option<Handle>,
    valid: impl Fn(Handle) -> bool,
) -> Result<Vec<Handle>, Error> {
    (0..batch)
        .map(|k| {
            (0..MAX_ATTEMPTS)
                .find_map(|attempt| candidate(k, attempt).filter(|&handle| valid(handle)))
                .ok_or(Error::Exhausted)
        })
        .collect()
}
