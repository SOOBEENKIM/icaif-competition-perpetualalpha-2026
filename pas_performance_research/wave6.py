"""Small declared multi-horizon timing ensemble, inventory-aware."""
import sys,json
from . import wave4
from .study import ROOT,BASE,sha,live_files
from .ensemble_timing import EnsembleTimingController
OUT=ROOT/'reports/pas_performance_research_20261022/wave6'
ARMS=[BASE,'pullback_any','pullback_both','pullback_majority']
def verify():
    for f,h in json.loads((OUT/'freeze.json').read_text()).items():assert sha(ROOT/f)==h,f
    assert live_files()==json.loads((OUT/'protocol.json').read_text())['live_files']
if __name__=='__main__':
    wave4.OUT=OUT;wave4.ARMS=ARMS;wave4.EntryController=EnsembleTimingController;wave4.verify=verify
    getattr(wave4,sys.argv[1])()
