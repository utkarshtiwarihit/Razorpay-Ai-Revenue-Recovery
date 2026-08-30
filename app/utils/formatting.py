"""
Indian-style currency formatting helpers.
We NEVER use "million" / "billion" anywhere in this project.
Big numbers use Lakh (L) and Crore (Cr); table/detail values use the
Indian comma-grouping style (e.g. 12,45,000).
"""


def inr_grouping(number: float) -> str:
    """Format a number using the Indian digit-grouping system.
    Example: 1234567 -> '12,34,567'
    """
    number = int(round(number))
    negative = number < 0
    number = abs(number)
    s = str(number)

    if len(s) <= 3:
        result = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        result = ",".join(parts) + "," + last3

    return f"-{result}" if negative else result


def format_inr(amount: float) -> str:
    """Full formatted rupee amount, e.g. ₹12,45,000"""
    return f"\u20b9{inr_grouping(amount)}"


def format_inr_compact(amount: float) -> str:
    """Compact rupee amount for KPI cards, using Lakh / Crore.
    Examples:
      45,000        -> ₹45.0 K
      12,45,000     -> ₹12.45 L
      1,23,45,000   -> ₹1.23 Cr
    """
    amount = float(amount)
    sign = "-" if amount < 0 else ""
    amount = abs(amount)

    if amount >= 1_00_00_000:  # 1 crore
        return f"{sign}\u20b9{amount / 1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:  # 1 lakh
        return f"{sign}\u20b9{amount / 1_00_000:.2f} L"
    elif amount >= 1_000:
        return f"{sign}\u20b9{amount / 1_000:.1f} K"
    else:
        return f"{sign}\u20b9{amount:.0f}"
