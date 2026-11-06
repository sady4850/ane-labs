#!/bin/zsh
# The standard sweep: about 1-3 hours. Results go to results/<chip>_macOS-<version>.jsonl.
cd "${0:A:h}/.."
make -s || exit 1
python3 -m anelab frontier ffn --functions 1,2,4,8,12,14,15,16,17,20,24,32 --max-units 64 --tag ffn
python3 -m anelab frontier ffn-plain --functions 8,15,16,17,24 --max-units 64 --tag ffn-plain
python3 -m anelab frontier matmul --functions 1,8,15,16,17,32 --max-units 128 --tag matmul
python3 -m anelab report
