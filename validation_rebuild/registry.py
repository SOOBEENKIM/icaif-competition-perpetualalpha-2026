"""Load the project's global registry without changing bundled trading imports."""
import importlib.util
from pathlib import Path
import sys

_path = Path(__file__).resolve().parents[1]/"replay_lab/evaluation_control/registry.py"
_name = "_sfinx_global_evaluation_registry"
if _name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _module
    _spec.loader.exec_module(_module)
else:
    _module = sys.modules[_name]
Registry = _module.Registry
EvaluationRejected = _module.EvaluationRejected
sha = _module.sha
