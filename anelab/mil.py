"""MIL programs for probing the ANE compiler.

`ffn` is a port of the FFN function of Splash's Neural Engine split
(runtime/ops/AneFfn.cpp, Apache-2.0), byte for byte for the same shape, so a
probe here compiles exactly what the engine compiles. The other families strip
it down to find which part of a program the compiler runs out of.
"""

import math
import struct
from dataclasses import dataclass

SEGMENT = 2560
BLOCK = 512
MAX_ROWS = 2048
INT8_UNIT = 128.0
INT8_PEAK = 127.0
FLOOR = 2.0**-9
CONSTANT_OFFSET = 64


def fp16(value):
    m, e = float(value).hex().split("p")
    m = m.rstrip("0").rstrip(".")
    return f"fp16({m}p{e})"


def dims(rows, width):
    return f"[1, 1, {rows}, {width}]"


def tensor(kind, rows, width):
    return f"tensor<{kind}, {dims(rows, width)}>"


@dataclass(frozen=True)
class Surface:
    rows: int
    width: int
    element: str

    @property
    def size(self):
        return 1 if self.element == "int8" else 2

    @property
    def stride(self):
        return (self.width * self.size + 63) // 64 * 64 // self.size

    def strides(self, rows):
        plane = rows * self.stride
        return f"[{plane}, {plane}, {self.stride}, 1]"

    def buffer(self, rows, width):
        return (f"tensor_buffer<{self.element}, shape={dims(rows, width)}, strides={self.strides(rows)}, "
                f"interleave_factors=[1, 1, 1, 1]>")


def constant_blob(count):
    data = 128
    blob = bytearray(data + 2 * count)
    struct.pack_into("<II", blob, 0, 1, 2)
    struct.pack_into("<IIQQ", blob, CONSTANT_OFFSET, 0xDEADBEEF, 1, 2 * count, data)
    return blob


def rotation_blob(channels, seed=20260930):
    import numpy as np

    signs = np.where(np.random.RandomState(seed).randint(0, 2**32, size=BLOCK, dtype=np.uint32) & 1, -1.0, 1.0)
    out = np.arange(BLOCK)
    parity = np.vectorize(lambda v: bin(v).count("1") & 1)(out[:, None] & out[None, :])
    hadamard = np.where(parity, -1.0, 1.0)
    block = (signs[:, None] * hadamard * np.float32(1.0 / math.sqrt(BLOCK))).astype(np.float16)
    values = np.tile(block, (channels // BLOCK, 1))
    blob = constant_blob(channels * BLOCK)
    blob[128:] = values.tobytes()
    return bytes(blob)


def segments(channels):
    return [min(SEGMENT, channels - b) for b in range(0, channels, SEGMENT)]


class Function:
    def __init__(self, name):
        self.name = name
        self.params = []
        self.body = []

    def line(self, text):
        self.body.append(f"        {text};\n")

    def f16(self, name, rows, width, expr):
        self.line(f"{tensor('fp16', rows, width)} {name} = {expr}")

    def take(self, name, surface, rows, width):
        self.params.append(f"{surface.buffer(rows, width)} {name}")
        self.line(f"{tensor(surface.element, rows, width)} {name}_t = tensor_buffer_to_tensor<ios17>(input = {name})")

    def give(self, name, value, surface, rows, width):
        self.line(f"{surface.buffer(rows, width)} {name} = tensor_to_tensor_buffer<ios17>(input = {value}, "
                  f"interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({surface.strides(rows)}))")
        return f"    func {self.name}<ios18>({', '.join(self.params)}) {{\n{''.join(self.body)}    }} -> ({name});\n"


def ffn(hidden, channels, rows, rotation=True):
    """Splash's AneFfn::function for an ANE share of `channels` intermediate channels."""
    unit = fp16(1.0 / INT8_UNIT)
    downs = segments(channels)
    f = Function(f"ffn{rows}")
    inputs = [(f"x{k}", Surface(SEGMENT, MAX_ROWS, "int8"), SEGMENT, rows) for k in range(hidden // SEGMENT)]
    inputs.append(("tx", Surface(1, MAX_ROWS, "fp16"), 1, rows))
    for k in range(hidden // SEGMENT):
        inputs.append((f"wg{k}", Surface(channels, SEGMENT, "int8"), channels, SEGMENT))
        inputs.append((f"wu{k}", Surface(channels, SEGMENT, "int8"), channels, SEGMENT))
    inputs.append(("sg", Surface(channels, 1, "fp16"), channels, 1))
    inputs.append(("su", Surface(channels, 1, "fp16"), channels, 1))
    for i, w in enumerate(downs):
        inputs.append((f"wd{i}", Surface(hidden, w, "int8"), hidden, w))
    inputs.append(("sd", Surface(hidden, 1, "fp16"), hidden, 1))
    for name, surface, r, w in inputs:
        f.take(name, surface, r, w)

    def matmul(name, weights, values, height, width):
        f.f16(f"{weights}_d", height, width, f"dequantize(input = {weights}_t, scale = {unit})")
        f.line(f"{tensor('fp16', height, rows)} {name} = matmul(transpose_x = bool(false), transpose_y = bool(false), "
               f"x = {weights}_d, y = {values})")

    def total(prefix, terms, height):
        acc = f"{prefix}0"
        for t in range(1, terms):
            nxt = f"{prefix}_sum{t}"
            f.f16(nxt, height, rows, f"add(x = {acc}, y = {prefix}{t})")
            acc = nxt
        return acc

    n = hidden // SEGMENT
    for k in range(n):
        f.f16(f"x{k}_d", SEGMENT, rows, f"dequantize(input = x{k}_t, scale = {unit})")
    for p in ("g", "u"):
        for k in range(n):
            matmul(f"{p}m{k}", f"w{p}{k}", f"x{k}_d", channels, SEGMENT)
        f.f16(f"{p}s", channels, rows, f"mul(x = {total(p + 'm', n, channels)}, y = s{p}_t)")
    c, r = channels, rows
    f.f16("gt", c, r, "mul(x = gs, y = tx_t)")
    f.f16("gh", c, r, f"mul(x = gt, y = {fp16(0.5)})")
    f.f16("th", c, r, "tanh(x = gh)")
    f.f16("tp", c, r, f"add(x = th, y = {fp16(1.0)})")
    f.f16("silu", c, r, "mul(x = gh, y = tp)")
    f.f16("h", c, r, "mul(x = silu, y = us)")
    if rotation:
        f.line(f"tensor<fp16, [1, {c}, 1, {r}]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, {c}, 1, {r}]))")
        f.line(f"tensor<fp16, [{c}, {BLOCK}, 1, 1]> rotation = const()[name = string(\"rotation\"), val = "
               f"tensor<fp16, [{c}, {BLOCK}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), "
               f"offset = uint64({CONSTANT_OFFSET})))]")
        f.line(f"tensor<fp16, [1, {c}, 1, {r}]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = "
               f"int32({c // BLOCK}), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), "
               f"strides = tensor<int32, [2]>([1, 1]), weight = rotation, x = h4)")
        f.f16("hr", c, r, f"reshape(x = hr4, shape = tensor<int32, [4]>({dims(c, r)}))")
    else:
        f.f16("hr", c, r, f"mul(x = h, y = {fp16(1.0)})")
    f.f16("habs", c, r, "abs(x = hr)")
    f.f16("peak", 1, r, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))")
    f.f16("floor", 1, r, f"maximum(x = peak, y = {fp16(FLOOR)})")
    f.f16("inverse", 1, r, f"real_div(x = {fp16(INT8_PEAK)}, y = floor)")
    f.f16("hs", c, r, "mul(x = hr, y = inverse)")
    f.line(f"{tensor('int8', c, r)} hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))")
    f.f16("hd", c, r, f"dequantize(input = hq, scale = {unit})")
    f.f16("hscale", 1, r, f"mul(x = floor, y = {fp16(INT8_UNIT / INT8_PEAK)})")
    begin = 0
    for i, w in enumerate(downs):
        f.f16(f"hd{i}", w, r, f"slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, {begin}, 0]), "
                              f"size = tensor<int32, [4]>({dims(w, r)}))")
        matmul(f"dm{i}", f"wd{i}", f"hd{i}", hidden, w)
        begin += w
    f.f16("ds", hidden, r, f"mul(x = {total('dm', len(downs), hidden)}, y = sd_t)")
    f.f16("yt", hidden + 1, r, "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))")
    return f.give("y", "yt", Surface(hidden + 1, MAX_ROWS, "fp16"), hidden + 1, r)


def matmul(k, n, rows, int8=False):
    """One matmul of weights [n, k] given as an input by rows [k, rows]."""
    f = Function(f"mm{rows}")
    kind = "int8" if int8 else "fp16"
    f.take("x", Surface(k, MAX_ROWS, "fp16"), k, rows)
    f.take("w", Surface(n, k, kind), n, k)
    w = "w_t"
    if int8:
        f.f16("w_d", n, k, f"dequantize(input = w_t, scale = {fp16(1.0 / INT8_UNIT)})")
        w = "w_d"
    f.line(f"{tensor('fp16', n, rows)} y_t = matmul(transpose_x = bool(false), transpose_y = bool(false), x = {w}, y = x_t)")
    return f.give("y", "y_t", Surface(n, MAX_ROWS, "fp16"), n, rows)


def program(functions):
    return "program(1.3)\n{\n" + "".join(functions) + "}\n"
