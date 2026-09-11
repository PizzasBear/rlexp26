//! Choosing which transitions a batch is made of.
//!
//! Not the read path as a whole: the walks over a buffer's slots -- the age band, the frame stack,
//! the n-step rollout -- are [`ReplayBuffer`]'s own methods, alongside the write path they read
//! after.
//! What lives here is only the policy on top of them. Given a buffer, propose a transition, check
//! it, redraw it if it fails, and hand back `batch_size` of them.
//!
//! It is the part with real alternatives -- rejection against a maintained index of samplable
//! slots, proportional against rank-based priorities -- so it is kept behind one entry point,
//! [`draw_batch`], and touches no storage: every slot it names is read back out of the buffer by
//! [`ReplayBuffer::sample`] afterwards.

use std::num::NonZero;

use rand::prelude::*;
use thiserror::Error;

use crate::{ReplayBuffer, ReplayBufferSampleParams};

/// A whole drawn batch, stored one column at a time.
///
/// Column-major because that is the shape the caller hands to Python: every one of these becomes
/// an array of its own, so a `Vec` of per-draw structs would only have to be taken back apart
/// field by field. Each column holds `batch_size` entries, bar the two stacks, which hold
/// `obs_stack` slots per draw, oldest frame first.
#[derive(Debug)]
pub struct DrawBatch {
    /// Flat `env * capacity + slot` indices, handed back to Python and to `update_prios`.
    pub indices: Vec<u32>,
    /// The priority each transition was drawn on, exactly as `update_prios` last stored it, or
    /// `1.0` throughout for a buffer sampling uniformly. Raw rather than normalised: dividing by
    /// `sum_prios()` recovers the proposal probability, but the caller may want the priority
    /// against `max_prio` instead, and cannot get back to it from a normalised number.
    pub prios: Vec<f32>,
    /// Discounted n-step returns.
    pub returns: Vec<f32>,
    /// Whether each rollout ended on a true episode end.
    pub terminals: Vec<bool>,
    pub obs_slots: Vec<u32>,
    pub next_obs_slots: Vec<u32>,
}

#[derive(Debug, Error, PartialEq)]
pub enum DrawError {
    #[error("no environment holds enough transitions to sample from yet")]
    TooSmall,
    #[error("every priority is zero, so there is nothing to sample")]
    NoPriorities,
    #[error("gave up finding a samplable transition after {MAX_ATTEMPTS} attempts")]
    Exhausted,
}

/// Attempts per draw before giving up.
///
/// The rejected band is `n_steps + obs_stack` slots out of an environment's whole capacity, so on
/// any realistic buffer a draw is accepted almost immediately. Reaching this cap means the
/// priorities have collapsed onto the write head -- a bug worth surfacing rather than spinning on.
const MAX_ATTEMPTS: u32 = 128;

/// Attempts a stratified draw spends inside its own stratum before widening to the whole mass.
///
/// A stratum is a contiguous slice of the priority mass, and the slots a draw has to reject are
/// contiguous in the tree as well: the `n_steps - 1` transitions behind a write head sit at
/// consecutive flat indices, all of them freshly written and so all carrying `max_prio`. A
/// stratum narrower than one of those runs can therefore contain nothing samplable at all, and
/// redrawing inside it will fail every time -- the retry budget above buys nothing, and the batch
/// dies on [`DrawError::Exhausted`].
///
/// That happens once `sum_prios / batch_size` drops below a run's worth of priority: a small
/// buffer, a batch comparable to the number of stored transitions, or a `max_prio` far above the
/// rest of the distribution. Widening costs that one draw its stratification -- its marginal
/// becomes the whole priority distribution instead of its slice of it, which is the ordinary
/// unstratified draw and still correctly proportional -- and leaves the rest of the batch alone.
/// Against failing the call outright that is the better trade, and in a batch that is not trapped
/// it never happens: acceptance on a healthy buffer is far above the first attempt.
const STRATIFIED_ATTEMPTS: u32 = 8;

/// The distribution [`draw_batch`] proposes from, prepared once for the whole batch.
///
/// Neither variant offers only samplable transitions -- the tree knows nothing of the age band,
/// and even a draw inside the band can land on a slot holding no complete transition -- so every
/// proposal is checked and redrawn if it fails.
///
/// It deliberately borrows nothing from the buffer, not even the tree it draws out of: each
/// proposal advances the buffer's own generator, so the buffer has to be free to be borrowed
/// mutably for the length of the draw.
enum Proposal {
    /// Proportional to stored priority, straight out of the sum tree.
    Prioritised { total: f32, stratified: bool },
    /// Uniform over every age in an environment's band. `cumulative` is the running count of those
    /// ages across environments, so one draw over `total` picks the environment and the age
    /// together without caring that they hold different numbers of them -- a mid-run `reset` costs
    /// one environment a slot without costing its neighbours one.
    Uniform { cumulative: Vec<u64>, total: u64 },
}

impl Proposal {
    /// Prepares the proposal, or reports why the buffer cannot serve a batch at all.
    fn new(rb: &ReplayBuffer, n_steps: u32) -> Result<Self, DrawError> {
        let mut cumulative = Vec::with_capacity(rb.num_envs());
        let mut valid_total = 0u64;
        for env in 0..rb.num_envs() {
            let band = rb.samplable_slot_ages(env, n_steps);
            valid_total += band.map_or(0, |band| u64::from(band.end() - band.start() + 1));
            cumulative.push(valid_total);
        }

        // Counted even for a prioritised draw: a tree full of priorities on slots no draw may land
        // on would otherwise spin to `Exhausted` rather than say what is actually wrong.
        if valid_total == 0 {
            return Err(DrawError::TooSmall);
        }

        let Some(prios) = rb.prios.as_ref() else {
            return Ok(Self::Uniform {
                cumulative,
                total: valid_total,
            });
        };

        let total = prios.total();
        if total <= 0.0 {
            return Err(DrawError::NoPriorities);
        }

        Ok(Self::Prioritised {
            total,
            stratified: rb.stratified,
        })
    }

    /// Proposes element `b` of the batch as `(env, slot, prio)`, or `None` for a point that landed
    /// outside the samplable band.
    ///
    /// `attempt` counts this element's rejected proposals so far, and only stratification reads
    /// it: past [`STRATIFIED_ATTEMPTS`] the draw widens from its own stratum to the whole mass.
    fn propose(
        &self,
        rb: &mut ReplayBuffer,
        n_steps: u32,
        b: usize,
        batch_size: usize,
        attempt: u32,
    ) -> Option<(usize, u32, f32)> {
        match self {
            Self::Prioritised { total, stratified } => {
                // Both scale a fraction of the total rather than drawing from a sub-range of it, so
                // that a total small enough for consecutive slice bounds to round together is still
                // a range to draw from.
                let fraction = if *stratified && attempt < STRATIFIED_ATTEMPTS {
                    (b as f32 + rb.rng.random::<f32>()) / batch_size as f32
                } else {
                    rb.rng.random::<f32>()
                };

                // The fraction is below one, but the product need not be below `total` once it has
                // rounded, and the tree's half-open range refuses `total` itself. The largest float
                // below it is what the top of a slice meant.
                let point = (fraction * total).min(total.next_down());
                let prios = (rb.prios.as_ref())
                    .expect("a prioritised proposal is only prepared for a buffer that has a tree");
                let (flat, prio) = prios.sample(point).ok()?;

                let capacity = rb.env_capacity();
                let (env, slot) = (flat / capacity, flat % capacity);

                // The tree carries a priority for every slot that ever held a transition, so a
                // slot the write head has since caught up with can come back and is refused here.
                let age = rb.slot_age(env, slot as _);
                rb.samplable_slot_ages(env, n_steps)?
                    .contains(&age)
                    .then_some((env, slot as _, prio))
            }
            Self::Uniform { cumulative, total } => {
                let r = rb.rng.random_range(0..*total);
                let env = cumulative.partition_point(|&c| c <= r);
                let band = rb
                    .samplable_slot_ages(env, n_steps)
                    .expect("an empty band holds no ages");
                let count = u64::from(band.end() - band.start() + 1);
                let age = band.start() + (r - (cumulative[env] - count)) as u32;

                // Every transition is equally likely, so every one carries the same priority, and
                // the importance-sampling weights it implies are all one -- which is correct: a
                // uniform draw needs no correction.
                Some((env, rb.slot_at_age(env, age), 1.0))
            }
        }
    }
}

/// One accepted draw, in the window between the attempt loop that produced it and the
/// [`DrawBatch`] columns it is appended to. Never collected: the batch is stored a column at a
/// time, so this only keeps the attempt loop's six results together on the way there.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Draw {
    env: usize,
    /// The drawn transition's slot, within its environment.
    slot: u32,
    /// The slot holding the observation its rollout ended on, `n_steps` later.
    next_slot: u32,
    prio: f32,
    ret: f32,
    terminal: bool,
}

/// Draws `batch_size` samplable transitions out of `rb`.
///
/// The proposal is the sum tree when the buffer keeps priorities and uniform over samplable ages
/// when it does not; the buffer's `stratified` applies only to the former, since spreading an
/// already uniform draw over the ages buys nothing. Either way the result is checked against the
/// age band, the slot type and the rollout, and redrawn if it fails. Rejection is cheaper here
/// than the alternative of keeping an exact index of samplable slots up to date on every
/// `save_step`. Its acceptance rate does not reach the caller: what comes back is the stored
/// priority, which rejection does not touch at all.
///
/// A stratified draw that keeps failing widens out of its stratum rather than spending the whole
/// retry budget somewhere nothing samplable lives; see [`STRATIFIED_ATTEMPTS`].
///
/// Every proposal, accepted or rejected, advances the buffer's own generator, so `rb` is taken
/// mutably: two draws of the same batch never see the same point.
pub fn draw_batch(
    rb: &mut ReplayBuffer,
    batch_size: usize,
    params: &ReplayBufferSampleParams,
) -> Result<DrawBatch, DrawError> {
    let capacity = rb.env_capacity();
    let obs_stack = rb.obs_stack.map_or(1, NonZero::get) as usize;
    let n_steps = params.n_steps.get();
    let discount = params.discount_or_one();
    let proposal = Proposal::new(rb, n_steps)?;

    let mut out = DrawBatch {
        indices: Vec::with_capacity(batch_size),
        prios: Vec::with_capacity(batch_size),
        returns: Vec::with_capacity(batch_size),
        terminals: Vec::with_capacity(batch_size),
        obs_slots: Vec::with_capacity(batch_size * obs_stack),
        next_obs_slots: Vec::with_capacity(batch_size * obs_stack),
    };

    for b in 0..batch_size {
        // Nothing reaches `out` until the attempts are over, so a rejection cannot leave a half
        // appended draw behind for the next one to write over.
        let mut draw = None;
        for attempt in 0..MAX_ATTEMPTS {
            let Some((env, slot, prio)) = proposal.propose(rb, n_steps, b, batch_size, attempt)
            else {
                continue;
            };
            let Some((ret, next_slot, terminal)) = rb.rollout(env, slot, n_steps, discount) else {
                continue;
            };

            draw = Some(Draw {
                env,
                slot,
                next_slot,
                prio,
                ret,
                terminal,
            });
            break;
        }
        let draw = draw.ok_or(DrawError::Exhausted)?;

        let base = draw.env * capacity;
        let start = out.obs_slots.len();
        out.obs_slots.resize(start + obs_stack, 0);
        out.next_obs_slots.resize(start + obs_stack, 0);

        rb.stack_from(draw.env, draw.slot, &mut out.obs_slots[start..]);
        rb.stack_from(draw.env, draw.next_slot, &mut out.next_obs_slots[start..]);

        // Flat, so the gather needs no separate environment index.
        let written = out.obs_slots[start..]
            .iter_mut()
            .chain(&mut out.next_obs_slots[start..]);
        for slot in written {
            *slot += base as u32;
        }

        out.indices.push(base as u32 + draw.slot);
        out.prios.push(draw.prio);
        out.returns.push(draw.ret);
        out.terminals.push(draw.terminal);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SlotType::{Final, Normal, Reset, Terminal};
    use crate::test_support::{one_env, params, spec};

    #[test]
    fn test_draw_never_returns_an_unsamplable_transition() {
        // Six slots, four written, head on slot 5. A two-frame stack leaves ages 1..=3, which are
        // slots 4, 3 and 2: slot 4 is an ordinary transition, slot 2 ends its episode, and slot 3
        // is the observation that ending parked -- the one draw that has to be rejected.
        let types = [Normal, Normal, Terminal, Final, Normal, Final];
        let mut rb = one_env(5, 4, &types, &spec().with_obs_stack(NonZero::new(2)));
        assert_eq!(rb.samplable_slot_ages(0, 1), Some(1..=3));

        let out = draw_batch(&mut rb, 200, &params(1)).unwrap();

        assert_eq!(out.indices.len(), 200);
        assert_eq!(out.obs_slots.len(), 400);
        assert_eq!(out.next_obs_slots.len(), 400);

        // Without priorities every transition is equally likely, so every one comes back at 1.0.
        assert!(out.prios.iter().all(|&prio| prio == 1.0));

        // Only the two acceptable slots ever come back, and only slot 2 is terminal.
        assert!(out.indices.iter().all(|&i| i == 2 || i == 4));
        assert!(out.indices.contains(&2));
        assert!(
            (out.indices.iter().zip(&out.terminals)).all(|(&i, &terminal)| terminal == (i == 2))
        );
    }

    #[test]
    fn test_a_batch_larger_than_the_buffer_still_draws() {
        // Three samplable transitions and a batch of sixty-four. Slicing the draws into
        // `batch_size` strata would leave most strata empty, which is not a range to draw from.
        let types = [Normal, Normal, Normal, Final, Reset, Reset];
        let mut rb = one_env(3, 3, &types, &spec());
        assert_eq!(rb.samplable_slot_ages(0, 1), Some(1..=3));

        let out = draw_batch(&mut rb, 64, &params(1)).unwrap();

        assert_eq!(out.indices.len(), 64);
        assert!(out.indices.iter().all(|&i| i < 3));
    }

    #[test]
    fn test_a_stratified_draw_survives_a_vanishing_priority_total() {
        // One priority, a single subnormal ulp. Sub-ranges of a total this small collapse onto
        // each other, so the stratified point has to be a fraction *of* the total, not a draw
        // from a slice of it.
        let types = [Normal, Normal, Normal, Final, Reset, Reset];
        let spec = spec().with_use_prios(true).with_stratified(Some(true));
        let mut rb = one_env(3, 3, &types, &spec);

        let prios = rb.prios.as_mut().expect("built with priorities");
        prios.update(2, f32::from_bits(1)).unwrap();

        let out = draw_batch(&mut rb, 32, &params(1)).unwrap();

        assert!(out.indices.iter().all(|&i| i == 2));
    }

    #[test]
    fn test_a_stratified_draw_widens_out_of_a_stratum_holding_nothing_samplable() {
        // Five written slots carrying equal priority, of which `n_steps = 3` leaves ages 1 and 2
        // -- slots 5 and 4 -- weighted but unsamplable. They are adjacent in the tree, so with
        // more strata than slots some stratum falls entirely inside that pair and every redraw
        // within it is rejected. The draw has to widen rather than exhaust its budget.
        let types = [Normal, Normal, Normal, Normal, Normal, Normal, Final];
        let spec = spec().with_use_prios(true).with_stratified(Some(true));
        let mut rb = one_env(6, 6, &types, &spec);
        assert_eq!(rb.samplable_slot_ages(0, 3), Some(3..=6));

        let prios = rb.prios.as_mut().expect("built with priorities");
        for slot in 0..6 {
            prios.update(slot, 1.0).unwrap();
        }

        let out = draw_batch(&mut rb, 64, &params(3)).expect("widens instead of exhausting");

        // Ages 3..=6 back from head 6 are slots 3, 2, 1 and 0; 4 and 5 are the trapped pair.
        assert_eq!(out.indices.len(), 64);
        assert!(out.indices.iter().all(|&i| i < 4), "{:?}", out.indices);
    }

    #[test]
    fn test_draw_reports_an_empty_buffer_rather_than_spinning() {
        // Freshly `reset`: one observation, no transitions.
        let mut rb = one_env(0, 0, &[Final, Reset, Reset, Reset], &spec());
        assert_eq!(rb.samplable_slot_ages(0, 1), None);

        assert_eq!(
            draw_batch(&mut rb, 8, &params(1)).unwrap_err(),
            DrawError::TooSmall,
        );
    }

    #[test]
    fn test_prioritised_draw_follows_the_priorities() {
        // Five written slots, all samplable with a one-frame stack, priority on only two of them.
        let types = [Normal, Normal, Normal, Normal, Normal, Final];

        // Rebuilt per pass rather than restratified in place, so that both draws start from the
        // same seeded generator and only the stratification differs.
        for stratified in [false, true] {
            let spec = spec()
                .with_use_prios(true)
                .with_stratified(Some(stratified));
            let mut rb = one_env(5, 5, &types, &spec);
            assert_eq!(rb.samplable_slot_ages(0, 1), Some(1..=5));

            let prios = rb.prios.as_mut().expect("built with priorities");
            prios.update(1, 1.0).unwrap();
            prios.update(3, 3.0).unwrap();

            let out = draw_batch(&mut rb, 4_000, &params(1)).unwrap();

            // Three times the priority, so roughly three quarters of the batch.
            let threes = out.indices.iter().filter(|&&i| i == 3).count();
            assert!(out.indices.iter().all(|&i| i == 1 || i == 3));
            assert!(
                (2_800..3_200).contains(&threes),
                "{threes} of 4000, stratified={stratified}"
            );

            // The priority comes back raw, not divided through by the tree's total.
            assert!(
                (out.indices.iter().zip(&out.prios))
                    .all(|(&i, &prio)| prio == if i == 3 { 3.0 } else { 1.0 })
            );
        }
    }
}
