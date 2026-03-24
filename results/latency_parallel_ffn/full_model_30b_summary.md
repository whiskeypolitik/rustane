# Full-Model FFN Latency Parallel Comparison

Config: `30B` — 5120d/13824h/96L/seq512

| mode | compile(s) | warmup(ms) | mean forward(ms) | tok/s | peak RSS(MB) | ANE hw(ms) | ANE busy % | avg layer(ms) | total FFN(ms) | total non-FFN(ms) | FFN % | max abs diff | mean abs diff | cosine | loss delta | win vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.87 | 23860.0 | 20782.3 | 24.64 | 118208 | 0.0 | 0.0% | 216.482 | 16178.5 | 4603.8 | 77.8% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +0.00% |
| shard2 | 0.69 | 14824.4 | 12981.5 | 39.44 | 198556 | 0.0 | 0.0% | 135.223 | 7222.9 | 5758.6 | 55.6% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +37.54% |
| shard4 | 0.75 | 13156.4 | 11194.1 | 45.74 | 204537 | 0.0 | 0.0% | 116.605 | 5372.6 | 5821.6 | 48.0% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +46.14% |
| shard6 | 0.95 | 11963.4 | 10357.2 | 49.43 | 198890 | 0.0 | 0.0% | 107.888 | 4586.5 | 5770.7 | 44.3% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +50.16% |
| shard8 | 0.97 | 11724.8 | 10111.4 | 50.64 | 198262 | 0.0 | 0.0% | 105.327 | 4331.4 | 5779.9 | 42.8% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +51.35% |
| shard12 | 1.29 | 11423.2 | 9855.4 | 51.95 | 198432 | 0.0 | 0.0% | 102.661 | 4077.2 | 5778.2 | 41.4% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +52.58% |
| shard16 | 1.44 | 11573.4 | 9980.3 | 51.30 | 198245 | 0.0 | 0.0% | 103.961 | 4195.8 | 5784.5 | 42.0% | 0.0000 | 0.0000 | 1.000000 | 0.000000 | +51.98% |

## Thresholds

- max abs diff <= `2.000e-2`
- mean abs diff <= `5.000e-3`
- cosine similarity >= `0.999`
- loss abs diff <= `1.000e-4`
- full forward win >= `5.0%`

Primary success: **PASS**

Winning modes: `shard2`, `shard4`, `shard6`, `shard8`, `shard12`, `shard16`

## Notes

- Each mode runs 1 warmup plus 2 measured stats-enabled full-model passes.
- Reported wall time, FFN totals, RSS, and ANE utilization all come from the same measured samples.
- This harness uses the mirrored stats-enabled path, not `full_model::forward_only_ws(...)`.
