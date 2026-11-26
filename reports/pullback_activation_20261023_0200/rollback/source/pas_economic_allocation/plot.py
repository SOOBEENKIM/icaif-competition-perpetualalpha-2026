"""Render exported numerical results without importing the trading environment."""
import json
import sys
from pathlib import Path
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=Path(sys.argv[1])
fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
for r in json.loads((folder/'plot_data.json').read_text()):
    x=[datetime.fromtimestamp(t,timezone.utc) for t in r['time']]
    axes[0].plot(x,r['pnl'],label=r['name'],color=r['color'])
    axes[1].plot(x,r['drawdown'],color=r['color'])
axes[0].set_ylabel('Net PnL (test USD)');axes[1].set_ylabel('Drawdown (%)');axes[0].legend()
axes[0].set_title('Reused reference-market history; conditional fills, not PAS performance')
for a in axes:a.grid(alpha=.2)
fig.autofmt_xdate();fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=140);plt.close(fig)
