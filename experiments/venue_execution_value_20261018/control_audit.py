"""Compatibility only for the old native order-log schema, never economics."""
import copy


def compare(fresh, old):
    normalized = copy.deepcopy(fresh)
    count = 0
    assert len(fresh['orders']) == len(old['orders'])
    for new_order, old_order in zip(normalized['orders'], old['orders']):
        if 'entry_limit_blocked' not in old_order and 'entry_limit_blocked' in new_order:
            assert new_order['entry_limit_blocked'] is False, 'Cannot erase a blocked order'
            del new_order['entry_limit_blocked']
            count += 1
    for field in ['metrics', 'events', 'orders', 'fills', 'owner_fills', 'snapshots', 'decisions']:
        assert normalized[field] == old[field], field
    return count
