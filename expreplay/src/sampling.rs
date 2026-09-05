//! The read path: choosing transitions and copying them out.
//!
//! Everything here is deliberately free of any `Python` token, so that the walks and the
//! arithmetic can be driven straight from `cargo test` -- the crate is an abi3 `cdylib` and links
//! no libpython, so a test can never construct a [`crate::ReplayBuffer`]. It is the read-path
//! counterpart of `EnvCursor` in `lib.rs`.

use std::ops::RangeInclusive;

use ndarray::prelude::*;
use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
use thiserror::Error;

use crate::SlotType;
use crate::prio_tree::PrioTree;

/// The ages a draw may land on in an environment that has written `len` slots, or `None` when it
/// cannot serve one yet.
///
/// Ages count back from the write head: age 0 is the head, age 1 the transition completed most
/// recently. An environment holds `min(len + 1, capacity)` observations -- one more than it has
/// transitions, since the head's observation is the previous transition's `next_obs` -- at ages
/// `0 ..= obs_len - 1`, and complete transitions at ages `1 ..= len`.
///
/// A draw at age `a` needs two things. Its frame stack reads backwards, ages
/// `a ..= a + obs_stack - 1`, which must stay inside the written observations rather than wrapping
/// past the oldest one onto the newest. Its rollout reads forwards, ages `a ..= a - n_steps + 1`,
/// which must stay on complete transitions, so `a >= n_steps`. The `next_obs` the rollout ends on
/// may be the head itself, at age 0, which is exactly what that observation is there for.
fn valid_ages(
    len: u32,
    capacity: u32,
    obs_stack: u32,
    n_steps: u32,
) -> Option<RangeInclusive<u32>> {
    let obs_len = (len + 1).min(capacity);
    let oldest = obs_len.checked_sub(obs_stack)?;

    (n_steps <= oldest).then_some(n_steps..=oldest)
}

/// Everything about a draw that is fixed for the whole batch.
///
/// Gathered into one value because `obs_stack` and `n_steps` between them decide which ages an
/// environment can serve, and both [`EnvSlots::new`] and [`draw_batch`] need them: passing them
/// separately to each would let the band be computed for one shape and the batch drawn for another.
#[derive(Debug, Clone, Copy)]
pub struct DrawSpec {
    /// Frames per observation, so also how far back a stack reads.
    pub obs_stack: u32,
    /// Steps a return accumulates over, so also how far forward a rollout reads.
    pub n_steps: u32,
    pub discount: f32,
    /// Whether a prioritised batch is spread over the priority mass rather than drawn
    /// independently. [`Proposal::Uniform`] ignores it -- every samplable transition is already
    /// equally likely -- and `ReplayBuffer` refuses the combination outright.
    pub stratified: bool,
}

/// Read-only view of one environment's slots, for the walks that `sample` does over them.
///
/// The write-path counterpart is `EnvCursor` in `lib.rs`; both exist so that the fiddly part of
/// each path can be tested without an interpreter.
pub struct EnvSlots<'a> {
    /// The slot the next observation will be written to.
    head: u32,
    /// Ages a draw may land on, per [`valid_ages`]. Carried here rather than in a parallel array
    /// so that a rejected proposal can be checked against its own environment's band directly.
    pub band: Option<RangeInclusive<u32>>,
    types: ArrayView1<'a, u8>,
    rewards: ArrayView1<'a, f32>,
}

impl<'a> EnvSlots<'a> {
    pub fn new(
        head: u32,
        len: u32,
        types: ArrayView1<'a, u8>,
        rewards: ArrayView1<'a, f32>,
        spec: DrawSpec,
    ) -> Self {
        let band = valid_ages(len, types.len() as u32, spec.obs_stack, spec.n_steps);

        Self {
            head,
            band,
            types,
            rewards,
        }
    }

    #[inline]
    fn capacity(&self) -> u32 {
        self.types.len() as u32
    }

    /// The slot `age` steps back from the write head.
    #[inline]
    pub fn at_age(&self, age: u32) -> u32 {
        (self.head + self.capacity() - age % self.capacity()) % self.capacity()
    }

    /// How far back from the write head a slot sits.
    #[inline]
    pub fn age(&self, slot: u32) -> u32 {
        (self.head + self.capacity() - slot) % self.capacity()
    }

    #[inline]
    fn ty(&self, slot: u32) -> SlotType {
        SlotType::from_raw(self.types[slot as usize])
    }

    #[inline]
    fn prev(&self, slot: u32) -> u32 {
        (slot + self.capacity() - 1) % self.capacity()
    }

    #[inline]
    fn next(&self, slot: u32) -> u32 {
        (slot + 1) % self.capacity()
    }

    /// Fills `out` with the slots holding the frames of the stack ending at `slot`, oldest first.
    ///
    /// The walk stops at an episode boundary and repeats the oldest frame it reached, which is
    /// what an environment's own frame-stack wrapper does at the start of an episode -- so the
    /// batch matches what the policy actually saw. A boundary is a `Final` slot: those hold an
    /// observation and no action, and every one of them is either the write head, which the age
    /// band keeps this walk away from, or the observation an ended episode was parked on, whose
    /// successor belongs to the next episode.
    pub fn stack_from(&self, slot: u32, out: &mut [u32]) {
        let (newest, rest) = out.split_last_mut().expect("a stack is at least one frame");
        *newest = slot;

        let mut slot = slot;
        for out in rest.iter_mut().rev() {
            let prev = self.prev(slot);
            if self.ty(prev) != SlotType::Final {
                slot = prev;
            }
            *out = slot;
        }
    }

    /// Rolls an `n_steps` return forward from `slot`, or rejects the draw.
    ///
    /// Returns the discounted return, the slot holding the observation the rollout ended on, and
    /// whether it ended on a true episode end -- so that nothing is bootstrapped past it.
    ///
    /// Rejects when `slot` holds no transition, and when the rollout meets a truncation with steps
    /// still to go: the caller applies one `discount ** n_steps` to the whole batch, so a return
    /// that stopped early would be bootstrapped with the wrong exponent -- unless nothing is
    /// bootstrapped at all, which is exactly the terminal case, and why that one is kept.
    pub fn rollout(&self, slot: u32, n_steps: u32, discount: f32) -> Option<(f32, u32, bool)> {
        let (mut ret, mut discounted, mut slot) = (0.0, 1.0, slot);

        for step in 0..n_steps {
            let ty = self.ty(slot);
            ret += discounted * self.rewards[slot as usize];
            discounted *= discount;
            slot = self.next(slot);

            match ty {
                SlotType::Normal => {}
                SlotType::Terminal => return Some((ret, slot, true)),
                SlotType::Truncated => return (step + 1 == n_steps).then_some((ret, slot, false)),
                // The write head and the observation parked after an episode end hold no reward.
                // Only a rollout's *first* slot can be one of those: every parked observation sits
                // directly behind a boundary, and the walk stops at every boundary. So this is a
                // draw to reject, not a buffer to distrust.
                SlotType::Final | SlotType::Reset => {
                    debug_assert_eq!(step, 0, "rollout walked into an incomplete slot");
                    return None;
                }
            }
        }

        Some((ret, slot, false))
    }
}

/// Copies one stored slot per row of `dst`, out of a `[env, slot, ..tail]` array.
///
/// `slots` holds flat `env * capacity + slot` indices, one per row, and `dst` is `[rows, ..tail]`.
/// A stacked output is gathered by reshaping its `[batch, stack, ..tail]` destination to
/// `[batch * stack, ..tail]` and passing `batch * stack` slots, oldest frame of each stack first:
/// because the stack axis leads, every frame is a contiguous run of bytes and each copy is a
/// memcpy rather than a stride-`stack` scatter.
pub fn gather_batch<T: Clone + Send + Sync + 'static>(
    dst: &mut ArrayRefD<T>,
    src: &ArrayRefD<T>,
    slots: &[u32],
    capacity: usize,
) {
    // A batch of stacked observations is worth spreading over the pool; a batch of one action
    // apiece is not. See [`crate::PAR_COPY_MIN_ELEMS`].
    if dst.len() < crate::PAR_COPY_MIN_ELEMS {
        ndarray::azip!((mut dst in dst.outer_iter_mut(), &flat in slots) {
            let (env, slot) = (flat as usize / capacity, flat as usize % capacity);
            dst.assign(&src.index_axis(Axis(0), env).index_axis(Axis(0), slot));
        });
    } else {
        ndarray::par_azip!((mut dst in dst.outer_iter_mut(), &flat in slots) {
            let (env, slot) = (flat as usize / capacity, flat as usize % capacity);
            dst.assign(&src.index_axis(Axis(0), env).index_axis(Axis(0), slot));
        });
    }
}

/// One accepted draw. The frame stacks live in [`DrawBatch`] alongside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Draw {
    /// Flat `env * capacity + slot` index, handed back to Python and to `update_prios`.
    pub index: u32,
    /// The priority this transition was drawn on, exactly as `update_prios` last stored it, or
    /// `1.0` for a buffer sampling uniformly. Raw rather than normalised: dividing by
    /// `sum_prios()` recovers the proposal probability, but the caller may want the priority
    /// against `max_prio` instead, and cannot get back to it from a normalised number.
    pub prio: f32,
    /// Discounted n-step return.
    pub ret: f32,
    /// Whether the rollout ended on a true episode end.
    pub terminal: bool,
}

/// A whole drawn batch: one [`Draw`] per element, plus the slots of its two frame stacks, oldest
/// frame first, `obs_stack` apiece.
#[derive(Debug)]
pub struct DrawBatch {
    pub draws: Vec<Draw>,
    pub obs_slots: Vec<u32>,
    pub next_obs_slots: Vec<u32>,
}

#[derive(Debug, Error, PartialEq)]
pub enum DrawError {
    #[error("no environment holds enough transitions to sample from yet")]
    TooSmall,
    #[error("every priority is zero, so there is nothing to sample")]
    NoPriorities,
    #[error(
        "gave up finding a samplable transition after {} attempts",
        MAX_ATTEMPTS
    )]
    Exhausted,
}

/// Attempts per draw before giving up.
///
/// The rejected band is `n_steps + obs_stack` slots out of an environment's whole capacity, so on
/// any realistic buffer a draw is accepted almost immediately. Reaching this cap means the
/// priorities have collapsed onto the write head -- a bug worth surfacing rather than spinning on.
const MAX_ATTEMPTS: u32 = 128;

/// A point in `[0, total)` for draw `b` of `batch_size`, proportional to priority.
///
/// A stratified draw takes its point from its own slice of the priority mass, so that a batch
/// spreads over the distribution instead of clumping; an i.i.d. one takes it from the whole of the
/// mass. The per-draw marginal is the same either way, only the draws' correlation differs.
fn prio_point(
    rng: &mut Xoshiro256PlusPlus,
    stratified: bool,
    b: usize,
    batch_size: usize,
    total: f32,
) -> f32 {
    // Both scale a fraction of the total rather than drawing from a sub-range of it, so that a
    // total small enough for consecutive slice bounds to round together is still a range to draw
    // from.
    let fraction = match stratified {
        true => (b as f32 + rng.random::<f32>()) / batch_size as f32,
        false => rng.random::<f32>(),
    };

    // The fraction is below one, but the product need not be below `total` once it has rounded,
    // and the tree's half-open range refuses `total` itself. The largest float below it is what
    // the top of a slice meant.
    (fraction * total).min(f32::from_bits(total.to_bits() - 1))
}

/// The distribution [`draw_batch`] proposes from, prepared once for the whole batch.
///
/// Neither variant offers only samplable transitions -- the tree knows nothing of the age band,
/// and even a draw inside the band can land on a slot holding no complete transition -- so every
/// proposal is checked and redrawn if it fails.
enum Proposal<'a> {
    /// Proportional to stored priority, straight out of the sum tree.
    Prioritised {
        prios: &'a PrioTree,
        total: f32,
        stratified: bool,
    },
    /// Uniform over every age in an environment's band. `cumulative` is the running count of those
    /// ages across environments, so one draw over `total` picks the environment and the age
    /// together without caring that they hold different numbers of them -- a mid-run `reset` costs
    /// one environment a slot without costing its neighbours one.
    Uniform { cumulative: Vec<u64>, total: u64 },
}

impl<'a> Proposal<'a> {
    /// Prepares the proposal, or reports why the buffer cannot serve a batch at all.
    fn new(
        envs: &[EnvSlots<'_>],
        prios: Option<&'a PrioTree>,
        stratified: bool,
    ) -> Result<Self, DrawError> {
        let mut cumulative = Vec::with_capacity(envs.len());
        let mut valid_total = 0u64;
        for env in envs {
            valid_total += env
                .band
                .as_ref()
                .map_or(0, |band| u64::from(band.end() - band.start() + 1));
            cumulative.push(valid_total);
        }

        // Counted even for a prioritised draw: a tree full of priorities on slots no draw may land
        // on would otherwise spin to `Exhausted` rather than say what is actually wrong.
        if valid_total == 0 {
            return Err(DrawError::TooSmall);
        }

        let Some(prios) = prios else {
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
            prios,
            total,
            stratified,
        })
    }

    /// Proposes element `b` of the batch as `(env, slot, prio)`, or `None` for a point that landed
    /// outside the samplable band.
    fn propose(
        &self,
        rng: &mut Xoshiro256PlusPlus,
        envs: &[EnvSlots<'_>],
        b: usize,
        batch_size: usize,
    ) -> Option<(usize, u32, f32)> {
        match self {
            Self::Prioritised {
                prios,
                total,
                stratified,
            } => {
                let point = prio_point(rng, *stratified, b, batch_size, *total);
                let (flat, prio) = prios.sample(point).ok()?;

                let capacity = envs[0].capacity();
                let (env, slot) = (flat / capacity as usize, flat as u32 % capacity);

                // The tree carries a priority for every slot that ever held a transition, so a
                // slot the write head has since caught up with can come back and is refused here.
                let age = envs[env].age(slot);
                envs[env]
                    .band
                    .as_ref()?
                    .contains(&age)
                    .then_some((env, slot, prio))
            }
            Self::Uniform { cumulative, total } => {
                let r = rng.random_range(0..*total);
                let env = cumulative.partition_point(|&c| c <= r);
                let band = envs[env]
                    .band
                    .as_ref()
                    .expect("an empty band holds no ages");
                let count = u64::from(band.end() - band.start() + 1);
                let age = band.start() + (r - (cumulative[env] - count)) as u32;

                // Every transition is equally likely, so every one carries the same priority, and
                // the importance-sampling weights it implies are all one -- which is correct: a
                // uniform draw needs no correction.
                Some((env, envs[env].at_age(age), 1.0))
            }
        }
    }
}

/// Draws `batch_size` samplable transitions.
///
/// The proposal is the sum tree when `prios` is set and uniform over samplable ages when it is
/// not; `stratified` applies only to the former, since spreading an already uniform draw over the
/// ages buys nothing. Either way the result is checked against the age band, the slot type and
/// the rollout, and redrawn if it fails. Rejection is cheaper here than the alternative of keeping
/// an exact index of samplable slots up to date on every `save_step`. Its acceptance rate does not
/// reach the caller: what comes back is the stored priority, which rejection does not touch at all.
pub fn draw_batch(
    rng: &mut Xoshiro256PlusPlus,
    envs: &[EnvSlots<'_>],
    prios: Option<&PrioTree>,
    batch_size: usize,
    spec: DrawSpec,
) -> Result<DrawBatch, DrawError> {
    let capacity = envs[0].capacity();
    let obs_stack = spec.obs_stack as usize;
    let proposal = Proposal::new(envs, prios, spec.stratified)?;

    let mut out = DrawBatch {
        draws: Vec::with_capacity(batch_size),
        obs_slots: Vec::with_capacity(batch_size * obs_stack),
        next_obs_slots: Vec::with_capacity(batch_size * obs_stack),
    };

    for b in 0..batch_size {
        let mut draw = None;
        for _ in 0..MAX_ATTEMPTS {
            let Some((env, slot, prio)) = proposal.propose(rng, envs, b, batch_size) else {
                continue;
            };
            let Some((ret, next_slot, terminal)) =
                envs[env].rollout(slot, spec.n_steps, spec.discount)
            else {
                continue;
            };

            // Only now that the draw is accepted: a rejection must leave `out` untouched.
            let base = env as u32 * capacity;
            let start = out.obs_slots.len();
            out.obs_slots.resize(start + obs_stack, 0);
            out.next_obs_slots.resize(start + obs_stack, 0);

            envs[env].stack_from(slot, &mut out.obs_slots[start..]);
            envs[env].stack_from(next_slot, &mut out.next_obs_slots[start..]);

            // Flat, so the gather needs no separate environment index.
            let written = out.obs_slots[start..]
                .iter_mut()
                .chain(&mut out.next_obs_slots[start..]);
            for slot in written {
                *slot += base;
            }

            draw = Some(Draw {
                index: base + slot,
                prio,
                ret,
                terminal,
            });
            break;
        }

        out.draws.push(draw.ok_or(DrawError::Exhausted)?);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use SlotType::{Final, Normal, Reset, Terminal};

    /// One environment's slot types, with a reward equal to the slot index so that a rollout's
    /// arithmetic is readable at a glance.
    fn env(types: &[SlotType]) -> (Array1<u8>, Array1<f32>) {
        let raw = types.iter().map(|&ty| ty as u8).collect();
        let rewards = (0..types.len()).map(|i| i as f32).collect();
        (raw, rewards)
    }

    /// A one-environment buffer, in the shape [`draw_batch`] takes.
    fn one_env<'a>(
        head: u32,
        len: u32,
        types: &'a Array1<u8>,
        rewards: &'a Array1<f32>,
        spec: DrawSpec,
    ) -> [EnvSlots<'a>; 1] {
        [EnvSlots::new(head, len, types.view(), rewards.view(), spec)]
    }

    /// The two fields the walks actually depend on; the rest only matter to [`draw_batch`].
    fn spec(obs_stack: u32, n_steps: u32) -> DrawSpec {
        DrawSpec {
            obs_stack,
            n_steps,
            discount: 0.99,
            stratified: false,
        }
    }

    #[test]
    fn test_valid_ages_leaves_room_for_the_stack_and_the_rollout() {
        // Eight slots written of sixteen, so nine observations exist, at ages 0..=8.
        // A four-frame stack ending at age 5 reads ages 5..=8, which is the oldest a draw can be.
        assert_eq!(valid_ages(8, 16, 4, 1), Some(1..=5));
        // A three-step rollout from age 3 reads ages 3, 2, 1 and lands its `next_obs` on the head.
        assert_eq!(valid_ages(8, 16, 4, 3), Some(3..=5));

        // Full: age 15 is the oldest slot, and its stack would wrap onto the head.
        assert_eq!(valid_ages(16, 16, 4, 1), Some(1..=12));
        assert_eq!(valid_ages(16, 16, 1, 1), Some(1..=15));

        // Nothing to draw yet: not enough frames for a stack, then not enough for the rollout.
        assert_eq!(valid_ages(2, 16, 4, 1), None);
        assert_eq!(valid_ages(4, 16, 4, 3), None);
        assert_eq!(valid_ages(0, 16, 1, 1), None);
    }

    #[test]
    fn test_ages_count_back_from_the_head_and_wrap() {
        let (types, rewards) = env(&[Normal, Final, Normal, Normal]);
        let slots = EnvSlots::new(1, 3, types.view(), rewards.view(), spec(1, 1));

        assert_eq!(slots.at_age(0), 1);
        assert_eq!(slots.at_age(1), 0);
        assert_eq!(slots.at_age(2), 3);
        assert_eq!(slots.at_age(3), 2);
        assert!((0..4).all(|age| slots.age(slots.at_age(age)) == age));

        assert_eq!(slots.band, Some(1..=3));
        assert_eq!(slots.ty(1), Final);
    }

    #[test]
    fn test_stack_from_walks_back_and_stops_at_an_episode_boundary() {
        // Slot 2 is the observation a terminated episode was parked on, so slot 3 begins a new
        // episode and nothing before slot 3 belongs to the same stack.
        let (types, rewards) = env(&[Normal, Terminal, Final, Normal, Normal, Normal, Final]);
        let slots = EnvSlots::new(6, 6, types.view(), rewards.view(), spec(4, 1));

        let mut out = [0; 4];

        // Three frames into the new episode: the walk reaches slot 3 and repeats it.
        slots.stack_from(5, &mut out);
        assert_eq!(out, [3, 3, 4, 5]);

        slots.stack_from(4, &mut out);
        assert_eq!(out, [3, 3, 3, 4]);

        // The first observation of the new episode has nothing behind it at all.
        slots.stack_from(3, &mut out);
        assert_eq!(out, [3, 3, 3, 3]);

        // The parked observation itself still belongs to the *old* episode, so the walk crosses
        // the terminal into it, and only stops at the write head on slot 6.
        slots.stack_from(2, &mut out);
        assert_eq!(out, [0, 0, 1, 2]);

        // A one-frame stack is just the slot.
        let mut one = [0; 1];
        slots.stack_from(4, &mut one);
        assert_eq!(one, [4]);
    }

    #[test]
    fn test_rollout_accumulates_until_a_boundary() {
        // Rewards are the slot index. Slot 3 ends the episode; slot 4 is the parked observation.
        let (types, rewards) = env(&[Normal, Normal, Normal, Terminal, Final, Normal, Final]);
        let slots = EnvSlots::new(6, 6, types.view(), rewards.view(), spec(1, 1));

        // A one-step return is just the slot's own reward, and `next_obs` is the slot after it.
        assert_eq!(slots.rollout(1, 1, 0.5), Some((1.0, 2, false)));

        // Slots 0, 1 and 2 are all `Normal`, so a three-step rollout from slot 0 runs its full
        // length: 0 + 0.5*1 + 0.25*2 = 1.0, ending its `next_obs` on slot 3.
        assert_eq!(slots.rollout(0, 3, 0.5), Some((1.0, 3, false)));

        // Reaching the terminal stops the rollout early, and the bootstrap is masked out.
        let expected = 1.0 + 0.5 * 2.0 + 0.25 * 3.0;
        assert_eq!(slots.rollout(1, 4, 0.5), Some((expected, 4, true)));

        // A rollout may not start on a slot that holds no transition.
        assert_eq!(slots.rollout(4, 1, 0.5), None);
        assert_eq!(slots.rollout(6, 1, 0.5), None);
    }

    #[test]
    fn test_rollout_rejects_a_truncation_it_cannot_reach_the_end_of() {
        use SlotType::Truncated;

        let (types, rewards) = env(&[Normal, Normal, Truncated, Final, Normal, Final]);
        let slots = EnvSlots::new(5, 5, types.view(), rewards.view(), spec(1, 1));

        // Landing on the truncation with the rollout's last step is fine: the caller's
        // `discount ** n_steps` is the right exponent, and `next_obs` is the successor to
        // bootstrap from.
        assert_eq!(slots.rollout(1, 2, 0.5), Some((1.0 + 0.5 * 2.0, 3, false)));

        // Reaching it with steps still to go is not: the return would be short but bootstrapped
        // as if it were not.
        assert_eq!(slots.rollout(1, 3, 0.5), None);
        assert_eq!(slots.rollout(2, 2, 0.5), None);
    }

    #[test]
    fn test_gather_pulls_slots_out_of_the_stored_layout() {
        // Stored as `[env=2, slot=3, 2]`, with the value encoding env/slot/element.
        let src = ArrayD::from_shape_fn(IxDyn(&[2, 3, 2]), |i| {
            (100 * i[0] + 10 * i[1] + i[2]) as u32
        });

        // Two frames per batch element, gathered through a merged `[batch * stack, 2]` view of a
        // `[batch, stack, 2]` output. Flat slots: env 0 slots 0 and 1, then env 1 slots 1 and 2.
        let mut dst = ArrayD::<u32>::zeros(IxDyn(&[2, 2, 2]));
        {
            let mut merged = dst
                .view_mut()
                .into_shape_with_order(IxDyn(&[4, 2]))
                .unwrap();
            gather_batch(&mut merged, &src, &[0, 1, 4, 5], 3);
        }
        let expected = vec![0, 1, 10, 11, 110, 111, 120, 121];
        assert_eq!(
            dst,
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), expected).unwrap()
        );

        // No stack: the destination is `[batch, 2]` and the same function serves.
        let mut dst = ArrayD::<u32>::zeros(IxDyn(&[2, 2]));
        gather_batch(&mut dst, &src, &[4, 0], 3);
        assert_eq!(
            dst,
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![110, 111, 0, 1]).unwrap()
        );
    }

    #[test]
    fn test_draw_never_returns_an_unsamplable_transition() {
        // Six slots, four written, head on slot 5. A two-frame stack leaves ages 1..=3, which are
        // slots 4, 3 and 2: slot 4 is an ordinary transition, slot 2 ends its episode, and slot 3
        // is the observation that ending parked -- the one draw that has to be rejected.
        let (types, rewards) = env(&[Normal, Normal, Terminal, Final, Normal, Final]);
        let envs = one_env(5, 4, &types, &rewards, spec(2, 1));
        assert_eq!(envs[0].band, Some(1..=3));

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let out = draw_batch(&mut rng, &envs, None, 200, spec(2, 1)).unwrap();

        assert_eq!(out.draws.len(), 200);
        assert_eq!(out.obs_slots.len(), 400);
        assert_eq!(out.next_obs_slots.len(), 400);

        // Without priorities every transition is equally likely, so every one comes back at 1.0.
        assert!(out.draws.iter().all(|d| d.prio == 1.0));

        // Only the two acceptable slots ever come back, and only slot 2 is terminal.
        assert!(out.draws.iter().all(|d| d.index == 2 || d.index == 4));
        assert!(out.draws.iter().any(|d| d.index == 2));
        assert!(out.draws.iter().all(|d| d.terminal == (d.index == 2)));
    }

    #[test]
    fn test_a_batch_larger_than_the_buffer_still_draws() {
        // Three samplable transitions and a batch of sixty-four. Slicing the draws into
        // `batch_size` strata would leave most strata empty, which is not a range to draw from.
        let (types, rewards) = env(&[Normal, Normal, Normal, Final, Reset, Reset]);
        let envs = one_env(3, 3, &types, &rewards, spec(1, 1));
        assert_eq!(envs[0].band, Some(1..=3));

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let out = draw_batch(&mut rng, &envs, None, 64, spec(1, 1)).unwrap();

        assert_eq!(out.draws.len(), 64);
        assert!(out.draws.iter().all(|d| d.index < 3));
    }

    #[test]
    fn test_a_stratified_draw_survives_a_vanishing_priority_total() {
        // One priority, a single subnormal ulp. Sub-ranges of a total this small collapse onto
        // each other, so the stratified point has to be a fraction *of* the total, not a draw
        // from a slice of it.
        let (types, rewards) = env(&[Normal, Normal, Normal, Final, Reset, Reset]);
        let envs = one_env(3, 3, &types, &rewards, spec(1, 1));

        let mut prios = PrioTree::new(6).unwrap();
        prios.update(2, f32::from_bits(1)).unwrap();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let out = draw_batch(
            &mut rng,
            &envs,
            Some(&prios),
            32,
            DrawSpec {
                stratified: true,
                ..spec(1, 1)
            },
        )
        .unwrap();

        assert!(out.draws.iter().all(|d| d.index == 2));
    }

    #[test]
    fn test_draw_reports_an_empty_buffer_rather_than_spinning() {
        // Freshly `reset`: one observation, no transitions.
        let (types, rewards) = env(&[Final, Reset, Reset, Reset]);
        let envs = one_env(0, 0, &types, &rewards, spec(1, 1));
        assert_eq!(envs[0].band, None);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        assert_eq!(
            draw_batch(&mut rng, &envs, None, 8, spec(1, 1)).unwrap_err(),
            DrawError::TooSmall,
        );
    }

    #[test]
    fn test_prioritised_draw_follows_the_priorities() {
        // Five written slots, all samplable with a one-frame stack, priority on only two of them.
        let (types, rewards) = env(&[Normal, Normal, Normal, Normal, Normal, Final]);
        let envs = one_env(5, 5, &types, &rewards, spec(1, 1));
        assert_eq!(envs[0].band, Some(1..=5));

        let mut prios = PrioTree::new(6).unwrap();
        prios.update(1, 1.0).unwrap();
        prios.update(3, 3.0).unwrap();

        for stratified in [false, true] {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
            let out = draw_batch(
                &mut rng,
                &envs,
                Some(&prios),
                4_000,
                DrawSpec {
                    stratified,
                    ..spec(1, 1)
                },
            )
            .unwrap();

            // Three times the priority, so roughly three quarters of the batch.
            let threes = out.draws.iter().filter(|d| d.index == 3).count();
            assert!(out.draws.iter().all(|d| d.index == 1 || d.index == 3));
            assert!(
                (2_800..3_200).contains(&threes),
                "{threes} of 4000, stratified={stratified}"
            );

            // The priority comes back raw, not divided through by the tree's total.
            let expected = |d: &Draw| if d.index == 3 { 3.0 } else { 1.0 };
            assert!(out.draws.iter().all(|d| d.prio == expected(d)));
        }
    }
}
