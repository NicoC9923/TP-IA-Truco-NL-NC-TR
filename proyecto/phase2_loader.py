"""Load a phase 2 champion with its sampling metadata."""
import json
from pathlib import Path

SELECTIONS = ('sampled', 'greedy')  # a sidecar without the field means 'sampled'


def load_phase2_metadata(weights_path, metadata=None):
    import math
    if metadata is None:
        try:
            metadata = json.loads((Path(weights_path).parent / 'reigning_champion.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError('champion metadata required') from exc
    if not isinstance(metadata, dict) or any(key not in metadata for key in ('temperature', 'profile_prior')):
        raise ValueError('temperature and profile_prior are required')
    for key in ('temperature', 'profile_prior'):
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if metadata.get('selection', 'sampled') not in SELECTIONS:
        raise ValueError("selection must be 'sampled' or 'greedy'")
    return metadata


def load_phase2_policy(weights_path, *, seed, adapt=True, metadata=None):
    metadata = load_phase2_metadata(weights_path, metadata)
    from phase2_torch import Phase2Network, Phase2Policy
    model = Phase2Network.load(weights_path)
    model.eval()
    return Phase2Policy(model, seed=seed, adapt=adapt, temperature=metadata['temperature'],
                        profile_prior=metadata['profile_prior'],
                        greedy=metadata.get('selection', 'sampled') == 'greedy')
