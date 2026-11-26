"""Multi-horizon entry timing and calibrated-prior exposure ablations."""
import sys,json
from . import wave4
from .study import ROOT,BASE,sha,live_files
from .soft_timing import SoftTimingController
OUT=ROOT/'reports/pas_performance_research_20261022/wave7'
ARMS=[BASE,'pullback_any3','pullback_any1','pullback_any_soft','pullback6_soft']
def verify():
    for f,h in json.loads((OUT/'freeze.json').read_text()).items():assert sha(ROOT/f)==h,f
    assert live_files()==json.loads((OUT/'protocol.json').read_text())['live_files']
if __name__=='__main__':
    wave4.OUT=OUT;wave4.ARMS=ARMS;wave4.EntryController=SoftTimingController;wave4.verify=verify
    getattr(wave4,sys.argv[1])()
