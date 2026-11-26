"""An asset forecast score cannot bypass the prior costed portfolio test."""
def economic_admission(candidate,baseline,minimum_days=30):
    c=candidate['metrics'];b=baseline['metrics']
    checks={
        'sufficient_complete_days':c['complete_days']>=minimum_days and b['complete_days']>=minimum_days,
        'candidate_engineering':all(candidate['gates'].values()),
        'baseline_engineering':all(baseline['gates'].values()),
        'positive_net_return':c['roi']>0,
        'net_return_improved':c['roi']>b['roi'],
        'daily_std_no_worse':c['daily_std_sample'] is not None and b['daily_std_sample'] is not None and c['daily_std_sample']<=b['daily_std_sample'],
        'mdd_no_worse':c['max_drawdown_magnitude']<=b['max_drawdown_magnitude'],
    }
    return {'admitted':all(checks.values()),'checks':checks,
            'candidate':c,'baseline':b,'decision_uses_only_prior_period':True}
