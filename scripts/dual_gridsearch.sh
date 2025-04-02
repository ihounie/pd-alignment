#!/usr/bin/env bash

for sr_tol in 0.1 0.01; do
  for rc in 1 10 100; do
    for sc in 1 10 100; do
      ./dual.sh --safety_ratio_tol "$sr_tol" --resilient_coeff "$rc" --scale_coeff "$sc"
    done
  done
done
