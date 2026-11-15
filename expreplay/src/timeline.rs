//! [`Timeline`]: where each environment's episodes begin and end, and which handles a draw may use.

use std::ops::Range;

use ndarray::prelude::*;

use crate::{EpisodeEnd, Error, Handle, Kind, Span};

/// What one timestep of the ring holds. Between calls a write head sits on an
/// [`Empty`](Self::Empty) or an [`Observed`](Self::Observed) slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Slot {
    /// Nothing yet: the head after an episode ended, waiting for the next one's first
    /// observation.
    Empty,
    /// An observation with no action: the head's, or the one an episode ended on.
    Observed,
    Step,
    Terminal,
    Truncated,
}

/// The episode structure of `num_envs` environments, each a ring of `env_capacity` timesteps,
/// under Gymnasium's next-step autoreset.
// One environment, `^` marking the write head. Episode one terminates at 3 and parks its final
// observation at 4; episode two is truncated at 8; episode three's newest observation is at 11.
//
// t:    0  1  2  3  4  5  6  7  8  9  10 11
// slot: S  S  S  Tm O  S  S  S  Tr O  S  O
//                                        ^
// S = Step, Tm = Terminal, Tr = Truncated, O = Observed.
pub struct Timeline {
    env_capacity: u64,
    /// Per environment: the timestep the next action is taken from, written or not.
    heads: Box<[u64]>,
    /// `[env, t % env_capacity]`, flattened.
    slots: Box<[Slot]>,
}

/// The stretch a valid draw reads, as [`Timeline::valid`] and [`Timeline::rollout`] both find it.
struct Reach {
    /// The oldest timestep of the stack, which stops at the start of the episode.
    first: u64,
    /// The real steps: short of the span's length only where the episode terminated.
    length: u32,
}

/// The timesteps a batch of draws read, one row per draw, for gathering out of any
/// [`Column`](crate::Column).
pub struct Rollout {
    /// `[batch, length]`: the transitions. Under [`EpisodeEnd::Pad`] a terminal repeats.
    pub steps: Array2<Handle>,
    /// `[batch, stack + length]`: one window of observations, oldest first. Observation `j`'s
    /// stack is `j .. j + stack`. It repeats the episode's first observation before its start
    /// and, under [`EpisodeEnd::Pad`], its final one past the end.
    pub frames: Array2<Handle>,
    /// `[batch, length]`
    pub kinds: Array2<Kind>,
    /// `[batch]`: the real steps, short of `length` only at a terminal.
    pub lengths: Array1<u32>,
}

impl Timeline {
    pub fn new(num_envs: usize, env_capacity: usize) -> Result<Self, Error> {
        if num_envs == 0 || env_capacity < 2 {
            return Err(Error::InvalidArgument(
                "num_envs must be at least one and env_capacity at least two".to_owned(),
            ));
        }
        let len = num_envs
            .checked_mul(env_capacity)
            .ok_or(Error::Allocation)?;

        Ok(Self {
            env_capacity: env_capacity as u64,
            heads: vec![0; num_envs].into_boxed_slice(),
            slots: vec![Slot::Empty; len].into_boxed_slice(),
        })
    }

    pub fn num_envs(&self) -> usize {
        self.heads.len()
    }

    pub fn env_capacity(&self) -> usize {
        self.env_capacity as usize
    }

    fn slot(&self, env: usize, t: u64) -> Slot {
        self.slots[env * self.env_capacity() + (t % self.env_capacity) as usize]
    }

    fn set_slot(&mut self, env: usize, t: u64, slot: Slot) {
        self.slots[env * self.env_capacity() + (t % self.env_capacity) as usize] = slot;
    }

    /// The timesteps of `env` still in the ring. Claiming a head overwrites the timestep one lap
    /// behind it, whether or not the head has been written yet.
    pub fn live(&self, env: usize) -> Range<u64> {
        let head = self.heads[env];
        let written = head + u64::from(self.slot(env, head) == Slot::Observed);

        (head + 1).saturating_sub(self.env_capacity)..written
    }

    /// The newest timestep `env` has written, or `None` before its first observation. A
    /// timestep's age is its distance behind this.
    pub fn newest(&self, env: usize) -> Option<u64> {
        self.live(env).end.checked_sub(1)
    }

    /// Starts an episode in every environment, filling `obs` with its first observation's handle.
    ///
    /// Mid-run this truncates the current episodes. One with no transition yet is overwritten,
    /// and its handle handed out again.
    pub fn reset(&mut self, obs: &mut [Handle]) -> Result<(), Error> {
        self.check_len("obs", obs.len())?;

        for (env, obs) in obs.iter_mut().enumerate() {
            let mut head = self.heads[env];
            if self.slot(env, head) == Slot::Observed
                && 0 < head
                && self.slot(env, head - 1) == Slot::Step
            {
                self.set_slot(env, head - 1, Slot::Truncated);
                head += 1;
                self.heads[env] = head;
            }

            self.set_slot(env, head, Slot::Observed);
            *obs = Handle {
                env: env as u64,
                t: head,
            };
        }

        Ok(())
    }

    /// Records one environment step each, filling `obs` with the incoming observation's handle.
    ///
    /// `stepped` says whether the call completed the transition at the timestep before `obs`. It
    /// is clear on the call after an episode's end, whose action the environment ignored.
    pub fn step(
        &mut self,
        terminated: &[bool],
        truncated: &[bool],
        obs: &mut [Handle],
        stepped: &mut [bool],
    ) -> Result<(), Error> {
        self.check_len("terminated", terminated.len())?;
        self.check_len("truncated", truncated.len())?;
        self.check_len("obs", obs.len())?;
        self.check_len("stepped", stepped.len())?;

        for env in 0..self.num_envs() {
            let (terminated, truncated) = (terminated[env], truncated[env]);
            let mut head = self.heads[env];

            stepped[env] = match self.slot(env, head) {
                Slot::Observed => {
                    let slot = match (terminated, truncated) {
                        (true, _) => Slot::Terminal,
                        (false, true) => Slot::Truncated,
                        (false, false) => Slot::Step,
                    };
                    self.set_slot(env, head, slot);
                    head += 1;
                    true
                }
                Slot::Empty => false,
                slot => unreachable!("head is {slot:?}, which no completed call leaves behind"),
            };

            self.set_slot(env, head, Slot::Observed);
            obs[env] = Handle {
                env: env as u64,
                t: head,
            };

            // That observation is the episode's last, and the next call's gets a timestep of its
            // own.
            if terminated || truncated {
                head += 1;
                self.set_slot(env, head, Slot::Empty);
            }
            self.heads[env] = head;
        }

        Ok(())
    }

    /// Whether a draw of `span` may start at `handle`: the one rule every sampler rejects by.
    ///
    /// The stack must be live back to its episode's start, and the rollout must fit below the
    /// write head. [`EpisodeEnd::Pad`] also wants a transition at `handle` and no truncation
    /// before the last step.
    pub fn valid(&self, handle: Handle, span: &Span) -> bool {
        self.reach(handle, span).is_some()
    }

    fn reach(&self, handle: Handle, span: &Span) -> Option<Reach> {
        let env = usize::try_from(handle.env)
            .ok()
            .filter(|&env| env < self.num_envs())?;
        let live = self.live(env);
        if !live.contains(&handle.t) {
            return None;
        }

        let mut first = handle.t;
        for _ in 1..span.stack() {
            // An episode's first observation follows the previous one's final observation, or
            // nothing at all.
            let Some(prev) = first.checked_sub(1) else {
                break;
            };
            if !live.contains(&prev) {
                return None;
            }
            if self.slot(env, prev) == Slot::Observed {
                break;
            }
            first = prev;
        }

        let length = match span.episode_end() {
            EpisodeEnd::Pad => {
                let mut length = span.length();
                for step in 0..span.length() {
                    // Every slot but the last is a `Step`, so the next timestep is written.
                    match self.slot(env, handle.t + u64::from(step)) {
                        Slot::Step => {}
                        Slot::Terminal => {
                            length = step + 1;
                            break;
                        }
                        Slot::Truncated if step + 1 == span.length() => {}
                        Slot::Truncated | Slot::Observed | Slot::Empty => return None,
                    }
                }
                length
            }
            EpisodeEnd::Continue => {
                let last = handle.t.checked_add(u64::from(span.length()))?;
                live.contains(&last).then_some(span.length())?
            }
        };

        Some(Reach { first, length })
    }

    /// The timesteps each draw of `span` reads. Fails with [`Error::Invalid`] on a start that
    /// [`valid`](Self::valid) refuses.
    pub fn rollout(&self, starts: &[Handle], span: &Span) -> Result<Rollout, Error> {
        let (batch, length, stack) = (starts.len(), span.length() as usize, span.stack() as usize);
        let mut steps = Vec::with_capacity(batch * length);
        let mut frames = Vec::with_capacity(batch * (stack + length));
        let mut kinds = Vec::with_capacity(batch * length);
        let mut lengths = Vec::with_capacity(batch);

        for &start in starts {
            let invalid = Error::Invalid {
                handle: start,
                span: *span,
            };
            let reach = self.reach(start, span).ok_or(invalid)?;
            let Handle { env, t } = start;
            let last_step = t + u64::from(reach.length) - 1;

            for step in 0..length as u64 {
                let t = (t + step).min(last_step);
                steps.push(Handle { env, t });
                kinds.push(match self.slot(env as usize, t) {
                    Slot::Step => Kind::Step,
                    Slot::Terminal => Kind::Terminal,
                    Slot::Truncated => Kind::Truncated,
                    Slot::Observed => Kind::Final,
                    Slot::Empty => unreachable!("a rollout stays below the write head"),
                });
            }
            for frame in 0..(stack + length) as u64 {
                let t = (t + frame).saturating_sub(stack as u64 - 1);
                frames.push(Handle {
                    env,
                    t: t.clamp(reach.first, last_step + 1),
                });
            }
            lengths.push(reach.length);
        }

        Ok(Rollout {
            steps: Array2::from_shape_vec((batch, length), steps).expect("length steps per draw"),
            frames: Array2::from_shape_vec((batch, stack + length), frames)
                .expect("stack + length frames per draw"),
            kinds: Array2::from_shape_vec((batch, length), kinds).expect("length kinds per draw"),
            lengths: Array1::from_vec(lengths),
        })
    }

    fn check_len(&self, name: &str, len: usize) -> Result<(), Error> {
        if len != self.num_envs() {
            return Err(Error::InvalidArgument(format!(
                "{name} must have one element per environment ({}), got {len}",
                self.num_envs()
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand::rngs::Xoshiro256PlusPlus;

    use super::*;

    /// One environment as a plain list of episodes, the definition the ring is checked against.
    #[derive(Default)]
    struct Model {
        episodes: Vec<Episode>,
    }

    struct Episode {
        first: u64,
        /// Observations so far. All but the last have an action.
        obs: u64,
        end: Option<Kind>,
    }

    impl Model {
        fn written(&self) -> u64 {
            self.episodes.last().map_or(0, |ep| ep.first + ep.obs)
        }

        fn open(&self) -> bool {
            self.episodes.last().is_some_and(|ep| ep.end.is_none())
        }

        fn live(&self, capacity: u64) -> Range<u64> {
            let head = self.written() - u64::from(self.open());
            (head + 1).saturating_sub(capacity)..self.written()
        }

        fn reset(&mut self) -> u64 {
            if self.open() {
                let ep = self.episodes.last_mut().unwrap();
                if ep.obs == 1 {
                    self.episodes.pop();
                } else {
                    ep.end = Some(Kind::Truncated);
                }
            }
            let first = self.written();
            self.episodes.push(Episode {
                first,
                obs: 1,
                end: None,
            });
            first
        }

        fn step(&mut self, terminated: bool, truncated: bool) -> (u64, bool) {
            if !self.open() {
                return (self.reset(), false);
            }
            let ep = self.episodes.last_mut().unwrap();
            ep.obs += 1;
            ep.end = match (terminated, truncated) {
                (true, _) => Some(Kind::Terminal),
                (false, true) => Some(Kind::Truncated),
                (false, false) => None,
            };
            (ep.first + ep.obs - 1, true)
        }

        fn kind(&self, t: u64) -> Option<Kind> {
            let ep =
                (self.episodes.iter()).find(|ep| (ep.first..ep.first + ep.obs).contains(&t))?;
            let p = t - ep.first;
            match ep.end {
                Some(_) if p + 1 == ep.obs => Some(Kind::Final),
                Some(end) if p + 2 == ep.obs => Some(end),
                None if p + 1 == ep.obs => None,
                _ => Some(Kind::Step),
            }
        }

        /// The steps, frames, kinds and length of a draw, by definition.
        #[allow(clippy::type_complexity)]
        fn rollout(
            &self,
            t: u64,
            span: &Span,
            capacity: u64,
        ) -> Option<(Vec<u64>, Vec<u64>, Vec<Kind>, u32)> {
            let live = self.live(capacity);
            if !live.contains(&t) {
                return None;
            }
            let ep =
                (self.episodes.iter()).find(|ep| (ep.first..ep.first + ep.obs).contains(&t))?;
            let (length, stack) = (u64::from(span.length()), u64::from(span.stack()));

            // A stack cut short by its episode's start reads that off the timestep before it,
            // which has to be live too.
            let back = (t - ep.first).min(stack - 1);
            let first = t - back;
            let oldest_read = if back < stack - 1 {
                first.saturating_sub(1)
            } else {
                first
            };
            if oldest_read < live.start {
                return None;
            }

            let real = match span.episode_end() {
                EpisodeEnd::Pad => {
                    let mut real = length;
                    for i in 0..length {
                        match self.kind(t + i)? {
                            Kind::Step => {}
                            Kind::Terminal => {
                                real = i + 1;
                                break;
                            }
                            Kind::Truncated if i + 1 == length => {}
                            Kind::Truncated | Kind::Final => return None,
                        }
                    }
                    real
                }
                EpisodeEnd::Continue => (t + length < self.written()).then_some(length)?,
            };

            let steps: Vec<u64> = (0..length).map(|i| t + i.min(real - 1)).collect();
            let frames = (0..stack + length)
                .map(|j| (t + j).saturating_sub(stack - 1).clamp(first, t + real))
                .collect();
            let kinds = steps.iter().map(|&t| self.kind(t).unwrap()).collect();

            Some((steps, frames, kinds, real as u32))
        }
    }

    #[test]
    fn test_timeline_agrees_with_a_list_of_episodes() {
        const NUM_ENVS: usize = 3;
        const CAPACITY: usize = 11;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let mut timeline = Timeline::new(NUM_ENVS, CAPACITY).unwrap();
        let mut models: [Model; NUM_ENVS] = Default::default();
        let mut spans = Vec::new();
        for length in [1, 2, 3, 6] {
            for stack in [1, 2, 4] {
                for episode_end in [EpisodeEnd::Pad, EpisodeEnd::Continue] {
                    spans.push(Span::new(length, stack, episode_end).unwrap());
                }
            }
        }

        let mut obs = [Handle { env: 0, t: 0 }; NUM_ENVS];
        let mut stepped = [false; NUM_ENVS];
        let (mut valid, mut ended) = (0, 0);

        for call in 0..1_000 {
            if call == 0 || rng.random_bool(0.02) {
                timeline.reset(&mut obs).unwrap();
                for (env, model) in models.iter_mut().enumerate() {
                    let t = model.reset();
                    assert_eq!(obs[env], Handle { env: env as u64, t });
                }
            } else {
                // The call after an episode's end carries flags the environment did not set.
                let flags = |rng: &mut Xoshiro256PlusPlus, p| {
                    std::array::from_fn(|env| models[env].open() && rng.random_bool(p))
                };
                let terminated: [bool; NUM_ENVS] = flags(&mut rng, 0.1);
                let truncated: [bool; NUM_ENVS] = flags(&mut rng, 0.1);

                (timeline.step(&terminated, &truncated, &mut obs, &mut stepped)).unwrap();
                for (env, model) in models.iter_mut().enumerate() {
                    let (t, completed) = model.step(terminated[env], truncated[env]);
                    assert_eq!(obs[env], Handle { env: env as u64, t });
                    assert_eq!(stepped[env], completed);
                }
            }

            for (env, model) in models.iter().enumerate() {
                assert_eq!(timeline.live(env), model.live(CAPACITY as u64));
                assert_eq!(timeline.newest(env), model.written().checked_sub(1));

                for t in model.written().saturating_sub(CAPACITY as u64 + 2)..model.written() + 2 {
                    let handle = Handle { env: env as u64, t };
                    for span in &spans {
                        let expected = model.rollout(t, span, CAPACITY as u64);
                        assert_eq!(
                            timeline.valid(handle, span),
                            expected.is_some(),
                            "call {call}, {handle:?}, {span:?}"
                        );
                        let Some((steps, frames, kinds, length)) = expected else {
                            timeline.rollout(&[handle], span).err().unwrap();
                            continue;
                        };

                        let rollout = timeline.rollout(&[handle], span).unwrap();
                        let ts = |handles: &Array2<Handle>| -> Vec<u64> {
                            assert!(handles.iter().all(|handle| handle.env == env as u64));
                            handles.iter().map(|handle| handle.t).collect()
                        };
                        assert_eq!(
                            ts(&rollout.steps),
                            steps,
                            "call {call}, {handle:?}, {span:?}"
                        );
                        assert_eq!(
                            ts(&rollout.frames),
                            frames,
                            "call {call}, {handle:?}, {span:?}"
                        );
                        assert_eq!(rollout.kinds.into_raw_vec_and_offset().0, kinds);
                        assert_eq!(rollout.lengths[0], length);

                        valid += 1;
                        ended += usize::from(length < span.length());
                    }
                }
            }
        }

        // The run has to have exercised both outcomes for the comparison to mean anything.
        assert!(
            10_000 < valid && 1_000 < ended,
            "{valid} valid, {ended} ended early"
        );
    }

    #[test]
    fn test_timeline_rejects_bad_arguments() {
        Timeline::new(0, 4).err().unwrap();
        Timeline::new(1, 1).err().unwrap();
        Span::new(0, 1, EpisodeEnd::Pad).unwrap_err();
        Span::new(1, 0, EpisodeEnd::Pad).unwrap_err();

        let mut timeline = Timeline::new(2, 4).unwrap();
        timeline.reset(&mut [Handle { env: 0, t: 0 }]).unwrap_err();

        let span = Span::new(1, 1, EpisodeEnd::Pad).unwrap();
        assert!(!timeline.valid(Handle { env: 2, t: 0 }, &span));
        assert_eq!(timeline.newest(0), None);
    }
}
