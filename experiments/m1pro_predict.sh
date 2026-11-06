#!/bin/zsh
# Points the fit N * (10.6 + S) <= 249 predicts but no sweep measured. Expected result after each.
cd "${0:A:h}/.."
while read n u want; do
  python3 -m anelab point ffn $n $u --tag predict && got=ok || got=fail
  echo "predict ffn $n functions $u units: want $want got $got"
done <<'EOF'
21 1 ok
22 1 fail
20 3 ok
18 11 ok
19 11 fail
18 15 ok
19 15 fail
15 25 ok
16 26 fail
15 26 ok
16 17 ok
16 25 fail
EOF
echo PREDICT DONE
