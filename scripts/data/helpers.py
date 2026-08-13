
import lance
from loguru import logger as logging


from omegaconf import DictConfig, OmegaConf
from pathlib import Path


def _lance_path(output: str, lance_name:str) -> str:
    """Derive the Lance dataset path inside the output folder."""
    p = Path(output)
    return str(p / (lance_name + '.lance'))


def _config_path(output: str, fname: str = None) -> Path:
    if fname is None:
        return Path(output) / f'config.yaml'
    else:
        return Path(output) / f'{fname}_config.yaml'
        


def _count_existing_episodes(output: str, lance_name:str) -> int:
    """Return number of episodes already in the Lance dataset, or 0 if absent."""
    lp = _lance_path(output, lance_name)
    if not Path(lp).exists():
        return 0
    try:
        col = lance.dataset(lp).to_table(columns=["episode_idx"]).column("episode_idx").to_pylist()
        return max(col) + 1 if col else 0
    except Exception:
        return 0


def _save_config(cfg: DictConfig, output: str, fname: str) -> None:
    config_path = _config_path(output, fname)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path)
    logging.info(f'Config saved → {config_path}')


def _check_config_compatibility(cfg: DictConfig, output: str) -> None:
    """Raise if a saved config exists and anything other than `episodes` changed."""
    config_path = _config_path(output)
    if not config_path.exists():
        return
    saved = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    current = OmegaConf.to_container(cfg, resolve=False)
    saved.pop('episodes', None)
    current.pop('episodes', None)
    if current != saved:
        all_keys = set(saved) | set(current)
        changed = [k for k in all_keys if saved.get(k) != current.get(k)]
        raise ValueError(
            f'Config mismatch on resume — changed keys: {changed}. '
            f'Use a different output path or delete the existing dataset to start fresh.'
        )

