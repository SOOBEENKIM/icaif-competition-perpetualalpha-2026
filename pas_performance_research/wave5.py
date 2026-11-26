"""Sealed timing grid; all development outcomes remain visible."""
import sys
from . import wave4
from .study import ROOT,BASE,sha,live_files
from .timing import TimingController
import json
OUT=ROOT/'reports/pas_performance_research_20261022/wave5'
ARMS=[BASE,'entry_pullback_1_0','entry_pullback_3_0','entry_pullback_12_0','entry_pullback_6_19','entry_pullback_3_19']
def verify():
    for f,h in json.loads((OUT/'freeze.json').read_text()).items():assert sha(ROOT/f)==h,f
    assert live_files()==json.loads((OUT/'protocol.json').read_text())['live_files']
if __name__=='__main__':
    wave4.OUT=OUT;wave4.ARMS=ARMS;wave4.EntryController=TimingController;wave4.verify=verify
    getattr(wave4,sys.argv[1])()
