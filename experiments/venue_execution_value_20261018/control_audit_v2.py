"""Compare economic paths while documenting a native-order schema extension.

The original account.py zeroes source_volume when a limit blocks execution;
account_v2.py zeroes capacity and additionally records entry_limit_blocked.
Only this added annotation may differ. All shared order fields remain exact.
"""
import copy


def compare(fresh, old):
    normalized = copy.deepcopy(fresh)
    counts = {'false': 0, 'true': 0}
    assert len(fresh['orders']) == len(old['orders'])
    for new_order, old_order in zip(normalized['orders'], old['orders']):
        if 'entry_limit_blocked' not in old_order and 'entry_limit_blocked' in new_order:
            blocked = new_order['entry_limit_blocked']
            assert type(blocked) is bool
            if blocked:
                assert new_order['filled_qty'] == old_order['filled_qty'] == 0
                assert new_order['capacity'] == old_order['capacity'] == 0
            counts[str(blocked).lower()] += 1
            del new_order['entry_limit_blocked']
    for field in ['metrics', 'events', 'orders', 'fills', 'owner_fills', 'snapshots', 'decisions']:
        assert normalized[field] == old[field], field
    return counts
