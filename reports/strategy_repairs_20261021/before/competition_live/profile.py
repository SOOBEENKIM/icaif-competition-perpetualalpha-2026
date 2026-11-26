"""Full team equity is 100k; the selected user strategy has a 50k sleeve."""
import hashlib
import json
from pathlib import Path
from scored_phase.profile import ROOT,BUNDLE,CANDIDATE,START,END,SYMBOLS,verify_bundle


def load_profile():
    p=json.loads((Path(__file__).parent/'config.json').read_text())
    if (p['dex'],p['collateral'],p['team_initial_capital'],p['initial_capital'],p['reserved_capital'],
            p['candidate'],p['start_unix'],p['end_unix']) != ('pas','USDP',100000.,50000.,50000.,CANDIDATE,START,END):
        raise ValueError('Wrong scored allocation or phase')
    if p['rest_url']!='https://api.hyperliquid-testnet.xyz' or p['allocation']['teammate_enabled']:
        raise ValueError('Unsupported venue or unconnected teammate strategy')
    return p


def verify_deployment():
    m=json.loads((Path(__file__).parent/'deployment_manifest.json').read_text())
    for name,digest in m['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Scored deployment changed: '+name)
    return m
