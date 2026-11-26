"""Isolated, time-limited PAR trial; the saved PAS deployment is unchanged."""
import json
import hashlib
from pathlib import Path
from scored_phase.profile import ROOT, BUNDLE, CANDIDATE, SYMBOLS, verify_bundle

END = 1792411200  # 2026-10-19 12:00 UTC / 21:00 KST


def verify_deployment():
    manifest=json.loads((Path(__file__).parent/'deployment_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Practice deployment changed: '+name)
    return manifest


def load_profile():
    p = json.loads((Path(__file__).parent/'config.json').read_text())
    if (p['dex'], p['collateral'], p['candidate'], p['initial_capital'], p['end_unix']) != (
            'par', 'USDPR', CANDIDATE, 1_000_000., END):
        raise ValueError('Wrong practice trial identity')
    if p['rest_url'] != 'https://api.hyperliquid-testnet.xyz' or p['collateral_token'] != 3032:
        raise ValueError('Wrong testnet collateral/endpoint')
    return p


START = json.loads((Path(__file__).parent/'config.json').read_text())['start_unix']
