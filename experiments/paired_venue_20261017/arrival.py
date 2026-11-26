"""Use only a book received after the simulated order reaches the venue."""
from experiments.execution_retry_20261016.engine import Paper


class ArrivalPaper(Paper):
    def apply_pending(self, now, books):
        ready, waiting = [], []
        for order in self.pending:
            if now < order['arrives']:
                waiting.append(order); continue
            book = books.get(order['coin'])
            if now > order['arrives'] + 5000:
                self.attempts.append({**order, 'status': 'unfilled_missing_arrival_observation'})
            elif book is None or book['received_ms'] < order['arrives'] or book['data']['time'] < order['submitted']:
                waiting.append(order)
            else:
                ready.append(order)
        self.pending = ready
        super().apply_pending(now, books)
        self.pending.extend(waiting)


def broker(metadata, scenario):
    protocol = dict(candidate=dict(attempts_per_coin_phase=1, retry_seconds=30, initial_quote_drift_bps=10),
                    common=dict(quote_limit_bps=10, asset_notional_limit=150000, minimum_order_notional=10))
    return ArrivalPaper('baseline', protocol, scenario, metadata)
