"""Compare published perp entry prices with the full confirmed-fill basis.

The captured PAS snapshots truncate entryPx at (6 - szDecimals) decimals,
then strip trailing zeros. For example2393.30770244 is published as2393.3
for ETH(szDecimals4). This is a presentation check, never a ledger rewrite.
Cash, fees, position quantities and the fill chain remain separate checks.
"""
from decimal import Decimal,InvalidOperation,ROUND_DOWN


def matches_reported_entry(reported,exact,sz_decimals):
    if not isinstance(sz_decimals,int) or not 0<=sz_decimals<=6:return False
    try:
        observed=Decimal(str(reported));basis=Decimal(str(exact))
        if not observed.is_finite() or not basis.is_finite() or min(observed,basis)<=0:return False
        unit=Decimal(10)**(sz_decimals-6)
        # Floating-point reconstruction can fall just below an exact grid edge.
        # The allowance is1e-8of a reporting unit, not a notional-dollar waiver.
        epsilon=unit*Decimal('1e-8')
        return observed in {(basis-epsilon).quantize(unit,rounding=ROUND_DOWN),
                            (basis+epsilon).quantize(unit,rounding=ROUND_DOWN)}
    except (InvalidOperation,ValueError,TypeError):return False
