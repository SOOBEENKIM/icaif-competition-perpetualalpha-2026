"""Show the selected profiles. Explicit verification: python -m strategy.verify."""
import json
from .agent import profiles

print(json.dumps(profiles(), ensure_ascii=False, indent=2))
