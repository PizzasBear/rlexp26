//! [`Uniform`]: every timestep a timeline holds, equally likely.

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;

use crate::sampling::fill;
use crate::{Error, Handle, Span, Timeline};

pub struct Uniform {
    rng: Xoshiro256PlusPlus,
}

impl Uniform {
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: match seed {
                Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
                None => rand::make_rng(),
            },
        }
    }

    /// Draws `batch` handles uniformly, with replacement, over those `span` can start at.
    pub fn sample(
        &mut self,
        batch: usize,
        timeline: &Timeline,
        span: &Span,
    ) -> Result<Vec<Handle>, Error> {
        // The running count of live timesteps across environments, so that one draw picks the
        // environment and the timestep together.
        let mut total = 0;
        let cumulative: Vec<u64> = (0..timeline.num_envs())
            .map(|env| {
                total += timeline.live(env).count() as u64;
                total
            })
            .collect();
        if total == 0 {
            return Err(Error::Exhausted);
        }

        fill(
            batch,
            |_, _| {
                let point = self.rng.random_range(0..total);
                let env = cumulative.partition_point(|&count| count <= point);
                let live = timeline.live(env);

                Some(Handle {
                    env: env as u64,
                    t: live.end - (cumulative[env] - point),
                })
            },
            |handle| timeline.valid(handle, span),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::EpisodeEnd;

    use super::*;

    #[test]
    fn test_uniform_covers_every_valid_handle_evenly() {
        // Environment 0 has run twice as long as environment 1 has room for.
        let mut timeline = Timeline::new(2, 8).unwrap();
        let (mut obs, mut done) = ([Handle { env: 0, t: 0 }; 2], [false; 2]);
        timeline.reset(&mut obs).unwrap();
        for step in 0..5 {
            let ended = [false, step == 2];
            timeline
                .step(&ended, &[false; 2], &mut obs, &mut done)
                .unwrap();
        }

        let span = Span::new(2, 1, EpisodeEnd::Pad).unwrap();
        let valid: Vec<Handle> = (0..2)
            .flat_map(|env| (0..8).map(move |t| Handle { env, t }))
            .filter(|&handle| timeline.valid(handle, &span))
            .collect();
        assert!(3 < valid.len());

        let mut uniform = Uniform::new(Some(0));
        let handles = uniform
            .sample(1_000 * valid.len(), &timeline, &span)
            .unwrap();
        for &expected in &valid {
            let count = handles.iter().filter(|&&handle| handle == expected).count();
            assert!(
                (850..1_150).contains(&count),
                "{expected:?} drawn {count} times"
            );
        }
        assert!(handles.iter().all(|handle| valid.contains(handle)));
    }

    #[test]
    fn test_uniform_gives_up_on_an_empty_timeline() {
        let timeline = Timeline::new(1, 8).unwrap();
        let span = Span::new(1, 1, EpisodeEnd::Pad).unwrap();
        let mut uniform = Uniform::new(Some(0));

        assert!(matches!(
            uniform.sample(1, &timeline, &span),
            Err(Error::Exhausted)
        ));
        assert!(uniform.sample(0, &timeline, &span).is_err());
    }
}
