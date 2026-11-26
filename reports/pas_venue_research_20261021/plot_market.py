"""Render with the system plotting environment, without changing live packages."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
aligned=json.loads((OUT/'synchronized_analysis.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'figure.facecolor':'#f6f8fc','axes.facecolor':'#f6f8fc'})
symbols=[r['symbol'] for r in aligned['markets']]
y=np.arange(len(symbols))
fig,axes=plt.subplots(1,2,figsize=(12,5.8),constrained_layout=True)
axes[0].barh(y-.18,[r['main_basis_bps']['median'] for r in aligned['markets']],
    height=.34,label='PAS mid vs mainnet reference',color='#315dba')
axes[0].barh(y+.18,[r['oracle_basis_bps']['median'] for r in aligned['markets']],
    height=.34,label='PAS mid vs PAS oracle',color='#d88924')
axes[0].set(yticks=y,yticklabels=symbols,xlabel='Median basis (bp)',title='Reference choice changes the apparent gap')
axes[0].invert_yaxis();axes[0].axvline(0,color='#555',lw=.6);axes[0].legend(loc='lower left',fontsize=8)
axes[1].barh(y,[r['spread_bps']['median'] for r in aligned['markets']],color='#315dba')
axes[1].axvline(6,color='#b54436',ls='--',label='Two maker fills: 6 bp fees')
axes[1].set(yticks=y,yticklabels=symbols,xlabel='Median quoted spread (bp)',title='Quoted spread is not net maker profit')
axes[1].invert_yaxis();axes[1].legend(loc='lower right',fontsize=8)
fig.suptitle('PAS public quotes | 2026-10-21 02:03-02:18 KST\nMatched server timestamps within 2 seconds; incomplete PAS capture',fontsize=12)
fig.savefig(OUT/'market_structure.png',dpi=170)
plt.close(fig)
