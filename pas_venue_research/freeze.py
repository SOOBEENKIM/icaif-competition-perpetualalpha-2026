import json
import time
import hashlib
from .capture import OUT,ROOT,save


def main():
    path=OUT/'protocol.json'
    if path.exists():raise FileExistsError('Never overwrite a frozen experiment')
    health=json.loads((OUT/'capture_health.json').read_text())
    now=time.time();start=(int(now)//60+2)*60
    config=dict(notional=250.,depth_fraction=.25,maker_fee_bps=3.,taker_fee_bps=9.,
        holding_seconds=60,order_ttl_seconds=30,latency_seconds=1.,cancel_latency_seconds=1.,
        book_max_age_seconds=15.,context_max_age_seconds=3.,
        basis_convergence_fraction=.5,exit_impact_bps=10.,entry_buffer_bps=5.,
        risk_z=1.645,maximum_asset_gross=500.,maximum_total_gross=2500.)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (ROOT/'pas_venue_research').glob('*.py')}
    save(path,dict(created_at=now,start=start,end=health['deadline'],config=config,
        models=['taker_basis','maker_basis'],control='no_new_trade_same_inventory',
        hypothesis='Partial convergence of local executable book to current PAS oracle, not to an old mainnet candle.',
        risk='Use pre-capture mainnet hourly volatility to reserve directional risk; no mainnet hedge is executed.',
        historical_independence=False,prospective_policy_frozen_before_start=True,
        reference_risk_fit_end=health['started'],code_hashes=sources,
        selection='No winner selection from this short diagnostic. A pass requires positive completed net PnL, '
                  'positive worst-scenario net PnL, >=30 nonoverlapping time blocks, zero residual/unresolved states '
                  'and a positive 95% time-block-bootstrap lower bound. Otherwise do not activate.',
        maker_fill='After arrival, use observed opposing-side trades only. Queue ahead is visible size at arrival; '
                   'do not credit cancellations. Trade-through is still limited to observed traded quantity. '
                   'Book touch alone never fills. Missing trade continuity invalidates fill claims.',
        terminal='Stop new entries early enough for TTL plus holding horizon. Pay observed reverse-book taker exit; '
                 'unfilled exit remains residual, never free liquidation.',
        limitations=['Short capture tests mechanics and costs, not durable profitability.',
                     'Counterfactual fills cannot prove queue position, own impact, or actual exchange fills.',
                     'Current oracle is a hypothesis anchor, not a guaranteed future liquidation price.',
                     'No price-gap strategy is riskless cross-venue arbitrage with nonconvertible test collateral.'],
        orders_sent=0,automatic_live_replacement=False))
    print(json.dumps(dict(start=start,end=health['deadline'],seconds=health['deadline']-start)),flush=True)


if __name__=='__main__':
    main()
