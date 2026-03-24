# Full-Model FFN Latency Parallel Comparison

Config: `50B` — 5120d/13824h/160L/seq512

| mode | compile(s) | warmup(ms) | median forward(ms) | tok/s | peak RSS(MB) | avg layer(ms) | total FFN(ms) | total non-FFN(ms) | FFN % | max abs diff | mean abs diff | cosine | loss delta | win vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.08 | 31299.3 | 29213.3 | 17.53 | 197010 | 182.583 | 28186.1 | 1027.2 | 96.5% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +0.00% |
| shard2 | 0.74 | 24281.7 | 21753.7 | 23.54 | 328261 | 135.960 | 12136.2 | 9617.4 | 55.8% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +25.54% |
| shard4 | 0.76 | 21686.8 | 18759.3 | 27.29 | 333846 | 117.246 | 8982.2 | 9777.1 | 47.9% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +35.79% |
| shard6 | 1.03 | 22093.7 | 17294.4 | 29.60 | 328201 | 108.090 | 7753.3 | 9541.1 | 44.8% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +40.80% |

## Thresholds

- max abs diff <= `2.000e-2`
- mean abs diff <= `5.000e-3`
- cosine similarity >= `0.999`
- loss abs diff <= `1.000e-4`
- full forward win >= `5.0%`

Primary success: **PASS**

Winning modes: `shard2`, `shard4`, `shard6`

## Notes

- Baseline wall time comes from `full_model::forward_only_ws(...)`.
- Baseline FFN totals come from a separate reference pass using `layer::forward_timed(...)` per layer.
- Sharded FFN totals come from the custom sharded full-model loop.
