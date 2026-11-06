# Findings

M1 Pro (16-core ANE), 32 GB, macOS 26.5 (25F71). Raw probes in `results/M1-Pro_macOS-26.5.jsonl`.
Width is in units of 512 ANE channels; S = ceil(channels / 2560) is the number of `down` segments of an `ffn` function.

## 1. Two different failures

| failure | when | error | time |
|---|---|---|---|
| shape | one function wider than 16384 channels (33 units), any number of functions | `ANECCompile() FAILED: err=( CompilationFailure )` | fails in 15-300 ms when the service is idle |
| budget | too many functions for their width | `ANECCompile() FAILED: err=( )` (empty) | fails after a full compile, 10-110 s |

The shape limit is on one dimension, not on the tensor's size (`experiments/shape_ops.py`, tag `shape-op` / `shape-dim`):

| one op over fp16 [channels, rows] | 16384 x 2048 | 16896 x 2048 | 16896 x 64 | 32768 x 64 |
|---|---|---|---|---|
| `mul` by 1 | ok | fail | fail | fail |
| `mul` by a per-channel scale, `tanh`, `reduce_max` over channels, `quantize`+`dequantize` | ok | fail | | |
| `slice_by_size` to 2560 channels | ok | ok | | |
| `matmul` [5120, channels] x [channels, 2048] | ok | fail | | |

So an elementwise op over more than 16384 channels fails, and so does a matmul whose inner dimension is that wide, however few rows (16896 x 64 is 1M elements, 16384 x 2048 is 33M and compiles). A `matmul` whose *output* has 65536 channels compiles (`matmul` family below), so the compiler splits that one itself.

## 2. The budget is functions x cost per function

`ffn` frontier (bisection over width, 1..64 units, for each number of functions):

| functions | widest ok | first fail |
|---:|---:|---:|
| 1, 2, 4, 8, 12, 14 | 32 | 33 (shape) |
| 15 | 30 | 31 |
| 16, 17 | 20 | 21 |
| 20 | 5 | 6 |
| 24, 32 | none | 1 |

Each edge sits where a function gets one more `down` segment (5->6 units is S 1->2, 20->21 is S 4->5, 30->31 is S 6->7), and inside one segment the width does not matter. All edges fit

    compiles  <=>  channels <= 16384  and  functions * (a + S) <= B,   a ~ 10.6, B ~ 249

that is, a function costs a fixed part of about 10.6 `down` segments (inputs, gate/up for 2 hidden segments, SwiGLU, requantize) plus one per segment, against one budget for the whole program. The most functions per width:

| S | units | max functions |
|---:|---:|---:|
| 1 | 1-5 | 21 |
| 2 | 6-10 | 19 |
| 3 | 11-15 | 18 |
| 4 | 16-20 | 17 |
| 5 | 21-25 | 15 |
| 6 | 26-30 | 15 |
| 7 | 31-32 | 14 |

The fit was made on the 12 frontier points above. Then 12 points it predicts, which no sweep had measured, were compiled (`experiments/m1pro_predict.sh`, tag `predict`): **12 of 12 as predicted**, 6 ok and 6 fail, including both sides of the edge at S = 1 (21 ok, 22 fail), S = 3 (18 ok, 19 fail, at 11 and at 15 units) and S = 6 (15 ok, 16 fail at 26 units).

| functions | units | S | N (a + S) | predicted | got |
|---:|---:|---:|---:|---|---|
| 21 | 1 | 1 | 244 | ok | ok |
| 22 | 1 | 1 | 255 | fail | fail |
| 20 | 3 | 1 | 232 | ok | ok |
| 18 | 11 | 3 | 245 | ok | ok |
| 19 | 11 | 3 | 258 | fail | fail |
| 18 | 15 | 3 | 245 | ok | ok |
| 19 | 15 | 3 | 258 | fail | fail |
| 15 | 25 | 5 | 234 | ok | ok |
| 16 | 26 | 6 | 266 | fail | fail |
| 15 | 26 | 6 | 249 | ok | ok |
| 16 | 17 | 4 | 234 | ok | ok |
| 16 | 25 | 5 | 250 | fail | fail |

The edge is tight: 15 x (10.6 + 6) = 249 compiles and 16 x (10.6 + 5) = 249.6 does not, so a and B are only known to about 1%.

The number of rows per function does not matter: at 23 units 15 functions of 1152..2048 rows (24000 rows in all) compile where 16 functions of 64..1920 (15424 rows) do not (tag `rows`), and Splash's 128..2048 and 256..2048 sets of 15 break at the same width (30 ok, 31 fails).

## 3. What does not count

- **The rotation constant and conv.** `ffn-plain` (no conv and no [channels, 512] fp16 constant, which is up to 16 MB) has the same frontier as `ffn` at 8, 15, 16, 17 and 24 functions.
- **The number of functions as such.** `matmul` (one int8 dequant + matmul per function) compiles 1, 8, 15, 16, 17 and 32 functions at 128 units (65536 channels). There is no fixed 15 or 16 function limit; light functions just use little of the budget.

## 4. For an engine

- Do not hardcode the function count. For an FFN split like Splash's, the number of functions the ANE takes depends on the ANE's share: `floor(249 / (10.6 + S))` on M1 Pro, 15 at Qwen3.8-27B's share 0.68 (23 units, S = 5), 17 up to 20 units, 21 up to 5 units.
- A failed compile costs a full compile time (up to ~110 s here) before the empty error. A fallback that retries with half the functions is cheap compared with losing the split for the process.
- The two failures can be told apart by the error: `CompilationFailure` for a shape (quick when the service is idle) vs an empty error for the budget (after a full compile).
- Keep any elementwise tensor at 16384 channels or fewer: Splash's split at share 0.8 of a model with an intermediate size above 20480 would hit it.

## Open

- What the budget is in the compiled program (ANE tasks? the `.hwx` size?). B and a are in "segment" units of this one family.
- Why `matmul` with a wide output escapes the 16384 limit, and whether an engine could split its wide elementwise ops the same way.
- Other chips and macOS versions: the budget probably differs per ANE generation.
