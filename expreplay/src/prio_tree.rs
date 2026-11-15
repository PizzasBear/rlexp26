//! [`PrioTree`]: one weight per timestep, drawn in proportion.

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;

use crate::column::check_len;
use crate::ring::Ring;
use crate::sampling::{STRATIFIED_ATTEMPTS, fill};
use crate::sum_tree::SumTree;
use crate::{Error, Handle, Span, Timeline};

/// A column of weights with a sum tree over it. A timestep with no weight written is never
/// drawn.
pub struct PrioTree {
    ring: Ring,
    tree: SumTree,
    rng: Xoshiro256PlusPlus,
}

impl PrioTree {
    pub fn new(num_envs: usize, env_capacity: usize, seed: Option<u64>) -> Result<Self, Error> {
        let ring = Ring::new(num_envs, env_capacity)?;
        let tree = SumTree::new(ring.len())?;

        Ok(Self {
            ring,
            tree,
            rng: match seed {
                Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
                None => rand::make_rng(),
            },
        })
    }

    pub fn num_envs(&self) -> usize {
        self.ring.num_envs()
    }

    pub fn env_capacity(&self) -> usize {
        self.ring.env_capacity()
    }

    /// The sum of every weight. A draw's probability is its weight over this.
    pub fn total(&self) -> f32 {
        self.tree.total()
    }

    /// Sets the weight at each handle, as [`Column::write`](crate::Column::write) does.
    ///
    /// Weights must be finite and non-negative, checked before anything is written.
    pub fn write(
        &mut self,
        handles: &[Handle],
        weights: &[f32],
        mut landed: Option<&mut [bool]>,
    ) -> Result<usize, Error> {
        check_len("weights", weights.len(), handles.len())?;
        if let Some(landed) = &landed {
            check_len("landed", landed.len(), handles.len())?;
        }
        if let Some(&weight) = weights.iter().find(|w| !(0.0 <= **w && w.is_finite())) {
            return Err(Error::InvalidWeight(weight));
        }

        let mut count = 0;
        for (i, (&handle, &weight)) in handles.iter().zip(weights).enumerate() {
            let claim = self.ring.claim(handle)?;
            if let Some(landed) = &mut landed {
                landed[i] = claim.is_some();
            }
            let Some((row, skipped)) = claim else {
                continue;
            };

            for skipped in skipped {
                self.tree.update(skipped, 0.0).expect("row in range");
            }
            self.tree
                .update(row, weight)
                .expect("row in range, weight checked");
            count += 1;
        }

        Ok(count)
    }

    /// The weight at each handle. Fails as [`Column::read`](crate::Column::read) does.
    pub fn read(&self, handles: &[Handle]) -> Result<Vec<f32>, Error> {
        (handles.iter())
            .map(|&handle| Ok(self.tree.prios()[self.ring.locate(handle)?]))
            .collect()
    }

    /// Draws `batch` handles with replacement, in proportion to weight among those `span` can
    /// start at on `timeline`, with each one's weight. `stratified` draws one handle from each
    /// of `batch` equal slices of the weight mass.
    pub fn sample(
        &mut self,
        batch: usize,
        timeline: &Timeline,
        span: &Span,
        stratified: bool,
    ) -> Result<(Vec<Handle>, Vec<f32>), Error> {
        let total = self.tree.total();
        if total <= 0.0 {
            return Err(Error::NoWeight);
        }

        let handles = fill(
            batch,
            |k, attempt| {
                // A fraction of the total, not a sub-range of it: at a tiny total, consecutive
                // slice bounds round together.
                let fraction = if stratified && attempt < STRATIFIED_ATTEMPTS {
                    (k as f32 + self.rng.random::<f32>()) / batch as f32
                } else {
                    self.rng.random::<f32>()
                };

                // The product can round up to `total`, which the half-open range refuses.
                let point = (fraction * total).min(total.next_down());
                let (row, _) = self.tree.sample(point).ok()?;
                self.ring.handle_at(row)
            },
            |handle| timeline.valid(handle, span),
        )?;
        let weights = self
            .read(&handles)
            .expect("drawn handles are live in the tree");

        Ok((handles, weights))
    }
}

#[cfg(test)]
mod tests {
    use crate::EpisodeEnd;

    use super::*;

    fn handle(env: u64, t: u64) -> Handle {
        Handle { env, t }
    }

    /// One environment, `steps` transitions into a single episode, and a tree weighting each.
    fn stepped(capacity: usize, steps: u64, weight: impl Fn(u64) -> f32) -> (Timeline, PrioTree) {
        let mut timeline = Timeline::new(1, capacity).unwrap();
        let mut tree = PrioTree::new(1, capacity, Some(0)).unwrap();
        let (mut obs, mut done) = ([handle(0, 0)], [false]);

        timeline.reset(&mut obs).unwrap();
        for _ in 0..steps {
            timeline
                .step(&[false], &[false], &mut obs, &mut done)
                .unwrap();
            let t = obs[0].t - 1;
            tree.write(&[handle(0, t)], &[weight(t)], None).unwrap();
        }

        (timeline, tree)
    }

    #[test]
    fn test_prio_tree_draws_in_proportion_to_weight() {
        let (timeline, mut tree) = stepped(8, 4, |t| (t + 1) as f32);
        let span = Span::new(1, 1, EpisodeEnd::Pad).unwrap();
        assert_eq!(tree.total(), 10.0);

        for stratified in [false, true] {
            let (handles, weights) = tree.sample(10_000, &timeline, &span, stratified).unwrap();
            let mut counts = [0; 4];
            for (handle, weight) in handles.iter().zip(weights) {
                assert_eq!(weight, (handle.t + 1) as f32);
                counts[handle.t as usize] += 1;
            }
            for (t, count) in counts.into_iter().enumerate() {
                let expected = 1_000 * (t as i32 + 1);
                let slack = if stratified { 1 } else { 150 };
                assert!(
                    (count - expected).abs() <= slack,
                    "{counts:?}, stratified {stratified}"
                );
            }
        }
    }

    #[test]
    fn test_prio_tree_keeps_only_what_the_timeline_accepts() {
        // The newest two transitions cannot start a three-step rollout, whatever they weigh.
        let (timeline, mut tree) = stepped(16, 6, |t| if t < 4 { 1.0 } else { 4.0 });
        let span = Span::new(3, 1, EpisodeEnd::Pad).unwrap();

        for stratified in [false, true] {
            let (handles, _) = tree.sample(64, &timeline, &span, stratified).unwrap();
            assert!(
                handles.iter().all(|handle| handle.t < 4),
                "stratified {stratified}"
            );
        }

        // With nothing valid the draw gives up, and with no weight it does not start.
        let span = Span::new(7, 1, EpisodeEnd::Pad).unwrap();
        assert!(matches!(
            tree.sample(1, &timeline, &span, false),
            Err(Error::Exhausted)
        ));
        let mut empty = PrioTree::new(1, 16, Some(0)).unwrap();
        assert!(matches!(
            empty.sample(1, &timeline, &span, false),
            Err(Error::NoWeight)
        ));
    }

    #[test]
    fn test_prio_tree_never_draws_a_row_at_its_previous_laps_weight() {
        // Twelve transitions through a ring of eight, but weights written only for the first
        // eight: rows 0..4 belong to timesteps 8..12 on the timeline and still weigh what
        // timesteps 0..4 did.
        let (mut timeline, mut tree) = stepped(8, 8, |_| 1.0);
        let (mut obs, mut done) = ([handle(0, 0)], [false]);
        for _ in 0..4 {
            timeline
                .step(&[false], &[false], &mut obs, &mut done)
                .unwrap();
        }

        let span = Span::new(1, 1, EpisodeEnd::Pad).unwrap();
        let (handles, _) = tree.sample(256, &timeline, &span, true).unwrap();
        assert!(
            handles.iter().all(|handle| (5..8).contains(&handle.t)),
            "{handles:?}"
        );

        // Writing past the skipped timesteps zeroes them rather than leaving them weighted.
        tree.write(&[handle(0, 11)], &[2.0], None).unwrap();
        assert_eq!(
            tree.read(&[handle(0, 8), handle(0, 10), handle(0, 11)])
                .unwrap(),
            [0.0, 0.0, 2.0]
        );
        assert_eq!(tree.total(), 6.0);
    }

    #[test]
    fn test_prio_tree_write_checks_before_it_writes() {
        let (_, mut tree) = stepped(4, 6, |_| 1.0);

        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            let result = tree.write(&[handle(0, 5), handle(0, 4)], &[3.0, bad], None);
            assert!(matches!(result, Err(Error::InvalidWeight(_))));
        }
        assert_eq!(tree.total(), 4.0);

        let mut landed = [true; 2];
        tree.write(
            &[handle(0, 1), handle(0, 5)],
            &[9.0, 3.0],
            Some(&mut landed),
        )
        .unwrap();
        assert_eq!(landed, [false, true]);
        assert_eq!(tree.total(), 6.0);
        assert!(matches!(tree.read(&[handle(0, 1)]), Err(Error::Dead(_))));
    }
}
