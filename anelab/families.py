"""Program families a probe compiles: each takes the number of functions and a width and gives (MIL, blob)."""

from . import mil

HIDDEN = 5120


def evenly(functions, top=mil.MAX_ROWS, step=64):
    """`functions` distinct row counts, multiples of `step`, spread evenly up to `top`."""
    rows = sorted({max(step, round(top * (i + 1) / functions / step) * step) for i in range(functions)})
    if len(rows) != functions:
        raise ValueError(f"{functions} functions do not fit distinct multiples of {step} up to {top}")
    return rows


def splash(minimum, functions=15, step=128, top=mil.MAX_ROWS):
    """Splash's functionRows(): the step's rows from `minimum`, then `top`, at most `functions` in all."""
    rows = []
    row = minimum
    while row < top and len(rows) + 1 < functions:
        rows.append(row)
        row += step
    return rows + [top]


def ffn(functions, units, rows=None, rotation=True, hidden=HIDDEN):
    rows = rows or evenly(functions)
    channels = units * mil.BLOCK
    src = mil.program([mil.ffn(hidden, channels, r, rotation=rotation) for r in rows])
    return src, (mil.rotation_blob(channels) if rotation else b"")


def ffn_plain(functions, units, rows=None):
    return ffn(functions, units, rows, rotation=False)


def matmul(functions, units, rows=None, k=HIDDEN, int8=True):
    rows = rows or evenly(functions)
    return mil.program([mil.matmul(k, units * mil.BLOCK, r, int8=int8) for r in rows]), b""


FAMILIES = {"ffn": ffn, "ffn-plain": ffn_plain, "matmul": matmul}
