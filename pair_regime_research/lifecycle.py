"""Use the same relationship/regime permission for entry, renewal and holding."""

def verdict(pair,index,state,now):
    if 'relationship_allowed' not in state:return dict(block_renewal=False,exit_reason=None)
    k=pair['pair_index'];source=int(state['signal_times'][index,k])
    if source>now:raise ValueError('Future formation/regime observation')
    if now-source>=3600:return dict(block_renewal=True,exit_reason=None,reason='stale_regime_source')
    relation=bool(state['relationship_allowed'][index,k]);regime=bool(state['regime_allowed'][index,k]);allowed=relation and regime
    reason=None if allowed else ('relationship_invalidated' if not relation else 'persistent_divergence')
    if source>pair.get('last_regime_source',-1):
        contiguous=source-pair.get('last_regime_source',source-3600)==3600
        prior=pair.get('regime_bad_sources',0) if contiguous else 0
        pair['regime_bad_sources']=0 if allowed else prior+1;pair['last_regime_source']=source
    return dict(block_renewal=not allowed,exit_reason=reason if pair.get('regime_bad_sources',0)>=2 else None,reason=reason,bad_sources=pair.get('regime_bad_sources',0))
