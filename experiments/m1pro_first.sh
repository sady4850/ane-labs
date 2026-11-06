#!/bin/zsh
cd "${0:A:h}/.."
S16=64,128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920
PR=128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,2048
BASE=256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,2048
for u in 20 21; do python3 -m anelab point ffn 16 $u --rows $S16 --tag repro; done
for u in 30 31; do
  python3 -m anelab point ffn 15 $u --rows $PR --tag repro
  python3 -m anelab point ffn 15 $u --rows $BASE --tag repro
done
python3 -m anelab frontier ffn --functions 1,2,4,8,12,14,15,16,17,20,24,32 --max-units 64 --tag ffn
python3 -m anelab frontier ffn-plain --functions 8,15,16,17,24 --max-units 64 --tag ffn-plain
python3 -m anelab frontier matmul --functions 1,8,15,16,17,32 --max-units 128 --tag matmul
echo DONE
