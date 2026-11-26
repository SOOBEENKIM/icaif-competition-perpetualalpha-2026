"""Research-only nominated controller. No live installation or order route."""
from .soft_timing import SoftTimingController

NAME='pullback_any3'

def controller(times,forecast,carry,anchors,protected,*,price_times,prices,age=None):
    """At a decision, source prices must be completed and known already.

    All ten markets and both signs remain eligible. For additions, buy at or
    below the3h OR24h earlier venue close, or sell at or above either close.
    Otherwise retain existing same-side quantity. Base exits and risk cuts
    always survive. This function cannot publish signals or place orders.
    """
    return SoftTimingController(times,forecast,carry,anchors,protected,5.,
        price_times=price_times,prices=prices,age=age,entry_rule=NAME)
