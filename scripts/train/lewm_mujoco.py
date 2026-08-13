"""Train LeWM on the CloudGripper MuJoCo tracking dataset.

Reuses lewm.py's training loop as-is (dataset-agnostic) with a mujoco-specific
Hydra config (scripts/train/config/lewm_mujoco.yaml -> data/cloudgripper_mujoco.yaml).

Usage:
    uv run python scripts/train/lewm_mujoco.py
    uv run python scripts/train/lewm_mujoco.py trainer.max_epochs=200
"""

import hydra

from lewm import train


@hydra.main(version_base=None, config_path='./config', config_name='lewm_mujoco')
def run(cfg):
    train(cfg)


if __name__ == '__main__':
    run()
