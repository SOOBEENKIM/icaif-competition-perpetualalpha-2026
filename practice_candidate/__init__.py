"""PAR scored-phase configuration and pre-deployment verification."""
from pathlib import Path
import sys

# Resolve inherited strategy modules from the audited candidate distribution.
# This also permits copying this directory beside that bundle for reproduction.
_source=Path(__file__).resolve().parents[1]/'deliverables/sfinx_pairs_candidate_20261018_final/source'
if _source.is_dir() and str(_source) not in sys.path:
    sys.path.insert(0,str(_source))
