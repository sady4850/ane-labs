"""Which op of the FFN function stops at 16384 channels: each alone over [channels, 2048] fp16 rows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anelab import mil, probe  # noqa: E402

ROWS = 2048


def single(op, c):
    f = mil.Function("f")
    f.take("x", mil.Surface(c, mil.MAX_ROWS, "fp16"), c, ROWS)
    out_rows = c
    if op == "copy":
        f.f16("y_t", c, ROWS, f"mul(x = x_t, y = {mil.fp16(1.0)})")
    elif op == "mul_rowscale":
        f.take("s", mil.Surface(c, 1, "fp16"), c, 1)
        f.f16("y_t", c, ROWS, "mul(x = x_t, y = s_t)")
    elif op == "tanh":
        f.f16("y_t", c, ROWS, "tanh(x = x_t)")
    elif op == "reduce_max":
        f.f16("y_t", 1, ROWS, "reduce_max(x = x_t, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))")
        out_rows = 1
    elif op == "quantize":
        f.line(f"{mil.tensor('int8', c, ROWS)} q = quantize(input = x_t, scale = fp16(1), output_dtype = string(\"int8\"))")
        f.f16("y_t", c, ROWS, f"dequantize(input = q, scale = {mil.fp16(1 / 128)})")
    elif op == "slice":
        f.f16("y_t", 2560, ROWS, f"slice_by_size(x = x_t, begin = tensor<int32, [4]>([0, 0, {c - 2560}, 0]), "
                                 f"size = tensor<int32, [4]>({mil.dims(2560, ROWS)}))")
        out_rows = 2560
    elif op == "down_matmul":
        f.take("w", mil.Surface(5120, c, "fp16"), 5120, c)
        f.line(f"{mil.tensor('fp16', 5120, ROWS)} y_t = matmul(transpose_x = bool(false), transpose_y = bool(false), "
               f"x = w_t, y = x_t)")
        out_rows = 5120
    return mil.program([f.give("y", "y_t", mil.Surface(out_rows, mil.MAX_ROWS, "fp16"), out_rows, ROWS)])


for op in ("copy", "mul_rowscale", "tanh", "reduce_max", "quantize", "slice", "down_matmul"):
    for c in (16384, 16896):
        r = probe.compile_program(single(op, c))
        probe.record({"tag": "shape-op", "family": f"op-{op}", "functions": 1, "units": c // 512, "channels": c,
                      **r})
        print(f"{op:<13} {c:>6}: {'ok' if r['ok'] else 'FAIL ' + probe.short_error(r).replace(chr(10), ' ')} "
              f"{r.get('compile_ms', '')} ms", flush=True)
