use thiserror::Error;

use crate::{AllocationError, try_zeroed_vec};

#[derive(Debug, Error)]
pub enum UpdateError {
    #[error("Supplied invalid priority value {0:?}")]
    InvalidPrio(f32),
    #[error("Failed to index priority tree")]
    IndexError,
}

#[derive(Debug, Error)]
#[error("Sample larger than tree size")]
pub struct SampleError(());

/// A sum-tree over `len` priorities, supporting O(log n) point update and O(log n) sampling
/// proportional to priority.
///
/// This is implemented as a Segment-Tree. While I considered using a Fenwick-Tree instead, I
/// realised the floating point errors would probably stack on top of each other. That intuition
/// holds up for a concrete reason: every internal node here is *re-derived* from its two children
/// on each update rather than adjusted by a delta, which makes the total a pairwise summation
/// whose relative error stays around `log2(len) * f32::EPSILON` no matter how many updates have
/// been applied. A Fenwick tree cannot re-derive like that -- it can only add the difference into
/// each covering node -- so its error really would accumulate without bound over a long run.
///
/// # Layout
///
/// `data` is an implicit binary heap of `2 * len - 1` nodes: the root is `0`, the children of `i`
/// are `2i + 1` and `2i + 2`, and each node holds the sum of its subtree. Since the length is
/// odd, node `i` has children exactly when `2i + 1 < data.len()`, i.e. when `i < offset()`, so the
/// last `len` nodes are the leaves and hold the priorities in order. This works for any `len`, not
/// only powers of two, and every non-root node has a sibling.
///
/// # Sampling invariant
///
/// [`sample`](Self::sample) descends maintaining `sample < subtree total`. That is what guarantees
/// it lands on a leaf with a *non-zero* priority, and it is why the accepted range is the
/// half-open `[0, total)` -- passing `total` itself would break the invariant at the first step
/// and could return an unset leaf.
pub struct PrioTree {
    data: Box<[f32]>,
}

impl PrioTree {
    pub fn new(len: usize) -> Result<Self, AllocationError> {
        if usize::MAX / 4 <= len {
            return Err(AllocationError::new());
        }
        let data_len = (2 * len).saturating_sub(1);
        let data = try_zeroed_vec(data_len)?.into_boxed_slice();

        Ok(Self { data })
    }

    /// Index of the first leaf, i.e. the number of internal nodes.
    #[inline]
    pub fn offset(&self) -> usize {
        self.data.len() / 2
    }

    /// Number of priorities the tree holds.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len().div_ceil(2)
    }

    /// Sum of every priority, and the exclusive upper bound for [`sample`](Self::sample).
    #[inline]
    pub fn total(&self) -> f32 {
        *self.data.first().unwrap_or(&0.)
    }

    /// The priorities themselves, in index order.
    #[inline]
    #[cfg(test)]
    pub fn prios(&self) -> &[f32] {
        &self.data[self.offset()..]
    }

    /// Sets the priority at `i`, repairing every node on the path back to the root.
    ///
    /// `value` must be finite and non-negative; zero is allowed and makes the entry unsamplable.
    pub fn update(&mut self, mut i: usize, value: f32) -> Result<(), UpdateError> {
        if self.len() <= i {
            return Err(UpdateError::IndexError);
        }
        if value < 0.0 || !value.is_finite() {
            return Err(UpdateError::InvalidPrio(value));
        }

        i += self.offset();
        self.data[i] = value;
        while 0 < i {
            let parent = (i - 1) / 2;
            let sibling = ((i + 1) ^ 1) - 1;
            self.data[parent] = self.data[i] + self.data[sibling];
            i = parent;
        }
        Ok(())
    }

    /// Maps a point in `[0, total())` to the index whose priority interval contains it, returning
    /// that index and its priority. Feeding this uniform noise samples proportional to priority.
    ///
    /// The half-open range is what keeps the descent off a zero-priority leaf, and it survives the
    /// rounding in the node totals: a `f32` strictly below a node's stored total is a full gap
    /// below it, which is at least as wide as half an ulp of either child, so `sample - left` can
    /// never round up as far as the right child's own total.
    pub fn sample(&self, mut sample: f32) -> Result<(usize, f32), SampleError> {
        if self.total() == 0.0 || !(0.0 <= sample && sample < self.total()) {
            return Err(SampleError(()));
        }

        let mut i = 0;
        loop {
            let left = 2 * i + 1;
            let Some(&left_total) = self.data.get(left) else {
                break;
            };

            if sample < left_total {
                i = left;
            } else {
                sample -= left_total;
                i = left + 1;
            }
        }

        Ok((i - self.offset(), self.data[i]))
    }
}

#[test]
fn test_prio_tree_basic() {
    // Test zero size works
    assert_eq!(PrioTree::new(0).unwrap().len(), 0);

    // Test sample, update & total
    let mut tree = PrioTree::new(3).unwrap();
    assert_eq!(tree.len(), 3);
    assert_eq!(tree.total(), 0.0);

    // Negative values are unsupported
    tree.update(0, -1.0).unwrap_err();
    assert_eq!(tree.total(), 0.0);

    // Negative zero is supported
    tree.update(0, -0.0).unwrap();
    assert_eq!(tree.total(), 0.0);

    // Positive values should work
    tree.update(1, 1.0).unwrap();
    assert_eq!(tree.total(), 1.0);

    tree.update(0, 2.0).unwrap();
    assert_eq!(tree.total(), 3.0);

    // This should be out of bounds
    tree.sample(3.0).unwrap_err();

    //    Current Tree
    // ===================
    //         3.0
    //     1.0     2.0
    //  1.0  0.0
    // ===================

    assert_eq!(tree.prios(), &[2.0, 1.0, 0.0]);
    assert_eq!(tree.data[..], [3.0, 1.0, 2.0, 1.0, 0.0]);

    assert_eq!(tree.sample(-0.0).unwrap(), (1, 1.0));
    assert_eq!(tree.sample(0.9).unwrap(), (1, 1.0));
    assert_eq!(tree.sample(1.0).unwrap(), (0, 2.0));
    assert_eq!(tree.sample(2.0).unwrap(), (0, 2.0));
    assert_eq!(tree.sample(2.9).unwrap(), (0, 2.0));
}

#[test]
fn test_prio_tree_layout_for_every_length() {
    // The `2 * len - 1` heap layout has to hold for lengths that are not powers of two, which is
    // where an off-by-one in `offset` would show up.
    for len in 1..=33 {
        let mut tree = PrioTree::new(len).unwrap();
        assert_eq!(tree.len(), len);
        assert_eq!(tree.prios().len(), len);

        for i in 0..len {
            tree.update(i, (i + 1) as f32).unwrap();
        }

        let expected = (1..=len).map(|x| x as f32).sum::<f32>();
        assert_eq!(tree.total(), expected, "wrong total for len {len}");
        assert!(
            (0..len).all(|i| tree.prios()[i] == (i + 1) as f32),
            "wrong leaves for len {len}"
        );
    }
}

#[test]
fn test_prio_tree_update_is_idempotent() {
    // Internal nodes are re-derived from their children, so writing the same value again must not
    // change anything. Accumulating into the parents instead would make the total drift upwards.
    let mut tree = PrioTree::new(8).unwrap();
    for i in 0..8 {
        tree.update(i, 1.0).unwrap();
    }
    assert_eq!(tree.total(), 8.0);

    for _ in 0..10 {
        tree.update(3, 1.0).unwrap();
        assert_eq!(tree.total(), 8.0);
    }

    // Lowering a priority has to lower the total, which a delta-free re-derivation gets for free.
    tree.update(3, 0.5).unwrap();
    assert_eq!(tree.total(), 7.5);
}

#[test]
fn test_prio_tree_sampling_is_proportional() {
    let mut tree = PrioTree::new(4).unwrap();
    for i in 0..4 {
        tree.update(i, (i + 1) as f32).unwrap();
    }
    assert_eq!(tree.total(), 10.0);

    // Sweeping uniformly across [0, total) must hit each leaf in proportion to its priority.
    const STEPS: usize = 10_000;
    let mut counts = [0usize; 4];
    for step in 0..STEPS {
        let point = (step as f32 + 0.5) / STEPS as f32 * tree.total();
        let (i, prio) = tree.sample(point).unwrap();
        assert_eq!(prio, tree.prios()[i]);
        counts[i] += 1;
    }
    assert_eq!(counts, [1000, 2000, 3000, 4000]);
}

#[test]
fn test_prio_tree_never_samples_a_zero_priority() {
    // Zeroed entries are the ones that have not been written yet. Returning one would hand out an
    // index into uninitialised transitions, so the descent invariant has to rule it out.
    let mut tree = PrioTree::new(16).unwrap();
    for i in [1, 7, 8, 15] {
        tree.update(i, 1.0).unwrap();
    }

    const STEPS: usize = 1_000;
    for step in 0..STEPS {
        let point = step as f32 / STEPS as f32 * tree.total();
        let (i, prio) = tree.sample(point).unwrap();
        assert!(prio > 0.0, "sampled unset index {i} at {point}");
        assert!([1, 7, 8, 15].contains(&i));
    }

    // The same, with priorities spread widely enough that the internal sums round: a tiny
    // priority next to a huge one is exactly where a node's total stops being the exact sum of
    // its children, and where a descent that trusted `<=` would fall off the end of a subtree.
    let mut tree = PrioTree::new(9).unwrap();
    for (i, prio) in [(0, 1e8), (3, 1e-8), (4, 1e8), (8, 1e-8)] {
        tree.update(i, prio).unwrap();
    }

    // Walk the largest representable points below `total`, which is where the accumulated
    // rounding in the node sums would show up first, and sweep the range as a whole.
    let mut point = tree.total();
    for _ in 0..STEPS {
        point = f32::from_bits(point.to_bits() - 1);
        let (i, prio) = tree.sample(point).unwrap();
        assert!(prio > 0.0, "sampled unset index {i} at {point}");
    }
    for step in 0..STEPS {
        let point = step as f32 / STEPS as f32 * tree.total();
        let (i, prio) = tree.sample(point).unwrap();
        assert!(prio > 0.0, "sampled unset index {i} at {point}");
    }
    // `total` is still excluded, however close the sum is to it.
    tree.sample(tree.total()).unwrap_err();
}

#[test]
fn test_prio_tree_rejects_invalid_input() {
    let mut tree = PrioTree::new(4).unwrap();

    // Priorities must be finite and non-negative, or the tree's sums stop meaning anything.
    tree.update(0, f32::NAN).unwrap_err();
    tree.update(0, f32::INFINITY).unwrap_err();
    tree.update(0, -1.0).unwrap_err();
    tree.update(4, 1.0).unwrap_err();
    assert_eq!(tree.total(), 0.0);

    // An empty distribution cannot be sampled from at all.
    tree.sample(0.0).unwrap_err();

    tree.update(0, 1.0).unwrap();
    tree.sample(-1.0).unwrap_err();
    tree.sample(f32::NAN).unwrap_err();
    // `total` itself is out of range: the interval is half-open.
    tree.sample(tree.total()).unwrap_err();
    tree.sample(tree.total() - 0.1).unwrap();
}

#[test]
fn test_prio_tree_error_stays_bounded_over_many_updates() {
    use rand::{RngExt, SeedableRng, rngs::Xoshiro256PlusPlus};

    // The reason for choosing a segment tree over a Fenwick tree: because each parent is recomputed
    // from its children, the root stays close to the true sum however many updates have gone
    // through it. This is the property that would degrade if `update` ever went back to
    // accumulating deltas.
    const LEN: usize = 100_000;
    let mut tree = PrioTree::new(LEN).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x243f_6a88_85a3_08d3);

    for i in 0..LEN {
        tree.update(i, rng.random()).unwrap();
    }
    for _ in 0..1_000_000 {
        let i = rng.random_range(0..LEN);
        tree.update(i, rng.random()).unwrap();
    }

    let reference: f64 = tree.prios().iter().map(|&p| p as f64).sum();
    let error = (tree.total() as f64 - reference).abs() / reference;
    assert!(
        error < 1e-5,
        "relative error {error:e} after 1M updates is larger than expected"
    );
}
