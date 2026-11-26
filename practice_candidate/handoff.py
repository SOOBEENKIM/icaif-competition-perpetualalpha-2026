"""Reconstruct existing practice inventory from actual exchange records."""
from decimal import Decimal
import numpy as np


def bootstrap(account, records, profile):
    if account.physical.fills or account.applied_fills:
        raise ValueError('Bootstrap only into a new ledger')
    transfers=records['userNonFundingLedgerUpdates']
    # This wallet has exactly one organizer grant. No external cash-flow
    # assumption is silently accepted if its history changes.
    if len(transfers)!=1:
        raise ValueError('Changed transfer history requires explicit reconciliation')
    grant=transfers[0]['delta']
    if (grant['type']!='spotTransfer' or grant['token']!='USDPR'
            or grant['destination'].lower()!=profile['team_wallet'].lower()
            or Decimal(grant['amount'])!=Decimal(str(profile['initial_capital']))):
        raise ValueError('Initial practice grant mismatch')
    for f in sorted(records['userFillsByTime'],key=lambda f:f['time']):
        if not f['coin'].startswith('par:'):continue
        identity=(f['coin'],f['tid'])
        if identity in account.applied_fills:continue
        j=account.physical.symbols.index(f['coin'].split(':')[1])
        if f['feeToken']!='USDPR' or f['side'] not in ('B','A'):
            raise ValueError('Unexpected historical fill currency/side')
        if abs(float(f['startPosition'])-account.physical.qty[j])>1e-8:
            raise ValueError('Incomplete or unordered fill history')
        q,px,fee=map(float,(f['sz'],f['px'],f['fee']))
        if not np.isfinite([q,px,fee]).all() or min(q,px)<=0:raise ValueError('Invalid fill')
        signed=q*(1 if f['side']=='B' else -1);rate=fee/(q*px)*10000
        before=account.physical.realized
        account.physical_fill(j,signed,px,rate,f['time']/1000,source='actual_handoff',tid=f['tid'])
        account.owner_fill('base',j,signed,px,rate,f['time']/1000,'actual_handoff')
        if abs(account.physical.realized-before-float(f['closedPnl']))>.01:
            raise ValueError('Historical realized PnL mismatch')
        account.applied_fills.add(identity)
    account.apply_actual_funding(records['userFunding'])
    return account
