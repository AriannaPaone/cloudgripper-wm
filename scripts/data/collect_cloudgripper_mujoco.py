"""Collect random trajectories from the CloudGripper MuJoCo simulation.

Usage:
    uv run python scripts/data/collect_cloudgripper_mujoco.py
"""

from pathlib import Path
import os

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy

import environments.mj_cloudgripper  # noqa: F401  (triggers gymnasium registration)


def main() -> None:

    root = Path(os.environ.get('STABLEWM_HOME', Path.home() / '.stable_worldmodel'))
    dataset_path = root / 'datasets' / 'cloudgripper_mujoco_random.lance'

    world = swm.World(
        "cloudgripper_mujoco/Tracking-v0",
        num_envs=4,
        image_shape=(64, 64),
        max_episode_steps=10,
    )
    world.set_policy(RandomPolicy(seed=0))

    world.collect(
        path=dataset_path,
        episodes=4,
        seed=0
    )
    world.close()
    

if __name__ == "__main__":
    main()
