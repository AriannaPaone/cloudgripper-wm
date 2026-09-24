"""Collect random trajectories from the CloudGripper MuJoCo simulation.

Usage:
    uv run python scripts/data/collect_cloudgripper_mujoco.py
"""

import hydra

import stable_worldmodel as swm
from loguru import logger as logging
from omegaconf import DictConfig
from hydra.utils import instantiate
from helpers import _lance_path, _count_existing_episodes, _check_config_compatibility, _save_config

import environments.mj_cloudgripper  # noqa: F401  (triggers gymnasium registration)


@hydra.main(version_base=None, config_path='./config', config_name='cloudgripper_mujoco')
def run(cfg: DictConfig) -> None:
    lance_out = _lance_path(cfg.output, cfg.output_name)

    n_existing = _count_existing_episodes(cfg.output, cfg.output_name)
    if n_existing > 0:
        _check_config_compatibility(cfg, cfg.output)
    to_collect = max(0, cfg.episodes - n_existing)

    if n_existing > 0:
        logging.info(
            f'Dataset exists: {n_existing} episodes. '
            f'Target: {cfg.episodes}. Collecting {to_collect} more.'
        )

    if to_collect == 0:
        logging.info('Target episode count already reached, nothing to collect.')
        return

    seed_start = cfg.seed + n_existing
    _save_config(cfg, cfg.output, cfg.output_name)

    world = swm.World(
        "cloudgripper_mujoco/Tracking-v0",
        num_envs=cfg.num_envs,
        image_shape=tuple(cfg.world.image_shape),
        max_episode_steps=cfg.world.max_episode_steps,
        max_delta=cfg.world.max_delta,
        height=cfg.world.height,
        width=cfg.world.width,
    )
    policy = instantiate(cfg.policy, seed=seed_start)
    world.set_policy(policy)

    # Optional MuJoCo viewer for live visualization of the environment.
    viewer_env = None
    viewer_opened = False
    collected = 0

    visualization_enabled = bool(cfg.get("visualization", {}).get("enabled", False))

    if visualization_enabled:
        if cfg.num_envs != 1:
            raise ValueError(
                "Visualization is only supported with num_envs=1."
            )
        # EnvPool contains the wrapped environments, we want the unwrapped CloudgripperMuJoCoEnv for the viewer.
        viewer_env = world.envs.envs[0].unwrapped

    try:

        if not visualization_enabled:
            while collected < to_collect:
                chunk = min(cfg.num_envs, to_collect - collected)
                seed = seed_start + collected
                if hasattr(policy, 'reset'):
                    policy.reset()
                world.collect(path=lance_out, episodes=chunk, seed=seed)
                collected += chunk
                logging.info(
                    f'Collected {n_existing + collected}/{cfg.episodes} episodes → {lance_out}'
                )
        else:
            # Teleop is 1 episode at a time
            while collected < to_collect:
                seed = seed_start + collected
                logging.info(
                    f'Starting teleop episode'
                    f'{collected + 1}/{to_collect} (seed={seed})'
                )
                if hasattr(policy, 'reset'):
                    policy.reset()

                #We have to reset the MuJoCo world
                world.reset(seed=seed)

                if not viewer_opened:
                    viewer_env.launch_passive_viewer(
                        show_left_ui=cfg.visualization.show_left_ui,
                        show_right_ui=cfg.visualization.show_right_ui,
                    )
                    viewer_opened = True
                    logging.info('Launched MuJoCo viewer for live visualization.')

                viewer_env.sync_passive_viewer()

                logging.info(
                    "Waiting for input. "
                    "W/S=X, A/D=Y, Q/E=Z, arrows=rotation/gripper, "
                    "SPACE=no-op, ESC=quit."
                )

                world.collect(
                    path=lance_out, 
                    episodes=1, 
                    seed=None,) #seed = None is important here because we want to keep the same seed for the episode, otherwise the environment will reset with a new seed and the teleop will be lost
                collected += 1
                logging.info(
                    f'Collected {n_existing + collected}/{cfg.episodes} episodes → {lance_out}'
                )
    except KeyboardInterrupt:
        logging.info('Keyboard interrupt received, stopping collection.')
    
    finally:
        if viewer_env is not None and viewer_opened:
            try:
                viewer_env.close_passive_viewer()
                logging.info('Closed MuJoCo viewer.')
            except Exception:
                pass
        world.close()
        logging.info(f'Collection finished. Total episodes collected: {n_existing + collected}/{cfg.episodes} → {lance_out}')
    

if __name__ == "__main__":
    run()
