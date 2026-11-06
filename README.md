# ane-labs

How big a program the Apple Neural Engine compiler takes, measured.

An engine that runs part of a model on the ANE compiles one MIL program with a fixed-shape function for each chunk size it wants to run (the ANE has no dynamic shapes). More functions means less padding for odd chunk sizes, but at some point `ANECCompile()` fails with an empty error (`err=( )`) and the whole program is gone. Nothing documents where that point is. This repo finds it by compiling programs of growing size through the same private `AppleNeuralEngine.framework` calls an engine uses, and records where they stop compiling.

It started from [paperniuk/splash#49](https://github.com/paperniuk/splash/pull/49), where a 16-function FFN program for Qwen3.8-27B did not compile on an M1 Pro and a 15-function one did.

## What is measured

A probe writes `model.mil` (+ `weights.bin`) into a temp dir, asks the ANE service to compile it (`-[_ANEClient compileModel:options:qos:error:]`), optionally to load it, then purges the compilation, so nothing stays in the service's cache. One process per probe. Result: ok or the failing stage and error, compile time.

Families of programs (`anelab/families.py`):

| family | each function | width |
|---|---|---|
| `ffn` | Splash's ANE FFN function: int8 gate/up/down matmuls with weights as inputs, SwiGLU, Hadamard rotation of the intermediate rows (grouped conv with a constant), per-token int8 requantize, hidden 5120 | ANE intermediate channels |
| `ffn-plain` | `ffn` without the rotation conv and its constant | same |
| `matmul` | one int8-dequant matmul, weights [channels, 5120] as an input | output channels |

Width is in units of 512 channels (Qwen3.8-27B has 34, Splash's share 0.68 is 23). Function `i` of `n` takes rows spread evenly up to 2048 (multiples of 64), unless given. `ffn` is a port of Splash's MIL generator, checked byte for byte against a program Splash compiled (same `model.mil` and `weights.bin`).

A frontier is, for each number of functions, the widest program that compiled and the narrowest that failed (bisection, assumes monotone in width).

## Run

```
make
python3 -m anelab point ffn 15 23                                   # one program
python3 -m anelab point ffn 15 23 --rows 128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,2048
python3 -m anelab frontier ffn --functions 8,15,16,17 --max-units 64
python3 -m anelab report
```

Needs Apple silicon, Xcode command line tools, Python 3 with numpy. Tested on macOS 26.5; the programs use MIL `ios18` functions, so older macOS may refuse them. Results go to `results/<chip>_macOS-<version>.jsonl`, one line per probe. A big program takes 30-90 s to compile and the CPU is busy meanwhile.

**Results from other Macs are very welcome**: run `experiments/frontier.sh` and send the jsonl (PR or issue).

## Findings

See [FINDINGS.md](FINDINGS.md).

## Caveats

- Private API. It can change with any macOS update; the probe only compiles (and loads with `--load`), never evaluates.
- The compiler's limit may depend on the macOS version as much as on the chip, so each result carries both.
- Only the families above. Other ops (attention, convs over images) may hit other limits.

## License

Apache-2.0. `anelab/mil.py` ports MIL text from Splash, see NOTICE.
