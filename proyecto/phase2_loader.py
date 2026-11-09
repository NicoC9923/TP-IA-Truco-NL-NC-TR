"""Load a phase 2 champion with its sampling metadata."""
import json
from pathlib import Path


def load_phase2_policy(weights_path, *, seed, adapt=True, metadata=None):
    if metadata is None:
        try:
            metadata = json.loads((Path(weights_path).parent / 'reigning_champion.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError('champion metadata required') from exc
    if not isinstance(metadata, dict) or any(key not in metadata for key in ('temperature', 'profile_prior')):
        raise ValueError('temperature and profile_prior are required')
    from phase2_torch import Phase2Network, Phase2Policy
    model = Phase2Network.load(weights_path)
    model.eval()
    return Phase2Policy(model, seed=seed, adapt=adapt, temperature=metadata['temperature'],
                        profile_prior=metadata['profile_prior'])
