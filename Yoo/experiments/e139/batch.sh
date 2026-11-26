#!/usr/bin/env bash
set -euo pipefail
out=${1:?supply a new result directory}
jobs=${2:-12}
exp=$(cd "$(dirname "$0")" && pwd)
py=/home/jio/fcsrl_src/.venv-skl191/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
export PYTHONPATH=/home/jio/fcsrl_src/results/e113/accel-deps
export out exp py
mkdir -p "$(dirname "$out")"
mkdir "$out"
git -C "$exp" rev-parse HEAD > "$out/commit"
printf '%s\n' 108 148 | xargs -P 2 -I@ bash -ec '"$py" "$exp/folds.py" prepare "$out" @ > "$out/prepare_@.log" 2>&1'
for seed in {10..14}; do
  for n in 108 148; do
    for method in fcsrl fcsrl_actor; do
      echo "$n $method $seed"
    done
  done
done | xargs -P "$jobs" -n3 bash -ec '"$py" "$exp/folds.py" train "$out" "$1" --method "$2" --seed "$3" > "$out/train_${1}_${2}_${3}.log" 2>&1' _
for seed in {10..14}; do
  for n in 108 148; do echo "$n $seed"; done
done | xargs -P "$jobs" -n2 bash -ec '"$py" "$exp/folds.py" evaluate "$out" "$1" --seed "$2" > "$out/eval_${1}_${2}.log" 2>&1' _
echo 'All training and evaluation runs completed.'
