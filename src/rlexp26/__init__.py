from itertools import count
from typing import Any

import ale_py
import gymnasium as gym
import jax
import numpy as np
import numpy.typing as npt
import optax
from flax import nnx
from gymnasium.spaces import MultiDiscrete
from gymnasium.vector import VectorEnv

gym.register_envs(ale_py)

from . import btr

NUM_ENVS = 64
ALE_PROTOCOL = dict[str, Any](
    repeat_action_probability=0.25,  # sticky actions (Machado et al. 2018)
    frameskip=4,
    stack_num=4,
    noop_max=30,
    use_fire_reset=True,
    episodic_life=False,  # True changes the task; report it if used
    reward_clipping=True,  # clip to [-1, 1] for training
    img_height=84,
    img_width=84,
)


def main() -> None:
    env: VectorEnv[npt.NDArray[np.uint8], npt.NDArray[np.uint64], npt.NDArray[Any]] = (
        gym.make_vec("ALE/Breakout-v5", num_envs=NUM_ENVS, **ALE_PROTOCOL)
    )

    assert isinstance(env.action_space, MultiDiscrete)

    num_actions: int = env.action_space.nvec[0]
    assert (env.action_space.nvec == num_actions).all()

    rngs = nnx.Rngs(0)
    qnet = btr.QNet(num_actions, rngs=rngs)
    target_qnet = nnx.clone(qnet)
    # TODO: maybe add a filter here to disable update_stats for SpectralNorm
    #       > target_qnet.set_attributes($filter(nnx.SpectralNorm), use_running_average=True)

    opt = nnx.Optimizer(
        qnet,
        optax.chain(
            optax.clip_by_global_norm(btr.GRADIENT_CLIPPING_MAX_NORM),
            optax.adam(btr.LEARNING_RATE, eps=btr.ADAM_EPS),
        ),
        wrt=nnx.Param,
    )

    obs, _info = env.reset()
    # TODO: Maybe apply this transformation through a wrapper of the environment
    obs = np.moveaxis(obs, -3, -1)

    for step in count():
        actions = btr.act(qnet, obs, rngs=rngs)
        actions.copy_to_host_async()

        # train step here

        actions = jax.device_get(actions)
        next_obs, reward, terminated, truncated, _info = env.step(actions)
        next_obs = np.moveaxis(next_obs, -3, -1)

        btr.opt_step(
            qnet=qnet,
            target_qnet=target_qnet,
            importance_sampling_weights=np.ones(NUM_ENVS),
            opt=opt,
            obs=obs,
            actions=actions,
            rewards=reward,
            dones=terminated,
            next_obs=next_obs,
            rngs=rngs,
        )

        if step % btr.TARGET_NETWORK_UPDATE_FREQ == 0:
            btr.update_target(qnet, target_qnet)

        print("done")

        obs = next_obs

        break
