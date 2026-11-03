# GLM long-context decode changes

The November 3 changes remove repeated preparation from GLM's sparse indexer and MTP paths. Their CPU correctness checks pass; real-pack CUDA qualification and throughput measurements are pending. The November 2 baseline remains the comparison point. These changes do not resolve the separate four-card DP gather bottleneck.

## Changed work

- A supported single-sequence cache owns a stable backend table of positions, pool membership, slots and representatives. Cache mutations upload patches instead of full per-evaluation facts, maps and bias arrays. Pool-map and bias outputs are derived on the backend. Mixed layer buffers, multi-sequence partitions, DP streams and unsupported backends retain the full reference path.
- Direct sparse indices are the default within the existing attention/layout bounds. The direct path filters and sorts physical cell IDs using the resident table, avoiding a full-width facts tensor and mask compaction. This preserves the existing membership rules, including tails, invalid slots and dump cells.
- MTP keeps selections in owned backend rows. Compatible mirrored tensors copy on their corresponding backends. The graph retains ownership across microbatch graph resets. A reused resident draft step skips unused scoring inputs and pooled-key compression; mutation, position and pool-boundary guards still invalidate reuse.
- Eligible isolated, single-head MTP histories calibrate adaptive depth from completed service cycles. Measurements include verification, catch-up and host work, without adding a synchronization. Cold, stale, mixed or interrupted histories retain the configured cost prior. Hidden-state batches also avoid reading a retained final row twice.

For a 400128-cell active extent, pool size four and 512 selected pools, the removed full-width host inputs total about 5.61 MB per card at one query row and 6.41 MB at three rows, plus dump scores. A one-cell append emits 32 bytes of canonical patches before a pool completes, or 56 bytes when it completes. Initial allocation, restore and rebuild can still require full initialization. Query parameters, tails and fresh compression inputs still transfer, and backend materialization still performs pool-wide device work. These byte counts are structural accounting, not a measured latency improvement.

## Reference controls

| Environment variable | Effect when set to `1` |
| --- | --- |
| `LLAMA_KPOOL_RESIDENT_LEGACY` | Use full host metadata inputs. |
| `LLAMA_KPOOL_DIRECT_INDICES_LEGACY` | Use the mask/compaction route. |
| `LLAMA_KPOOL_SELECTION_HOST_LEGACY` | Capture MTP selections through the host. |
| `LLAMA_MTP_TIMING_LEGACY` | Use the configured adaptive-cost prior without service-cycle calibration. |

The eight-row indexer MMA threshold, 64-row hidden-state threshold and one-row quantized gate-fusion boundary are retained. Earlier measurements found regressions when changing those paths. Inlining query rounding into every long-context scorer block would repeat that arithmetic per block; removing its separate launch alone does not establish a net improvement.

## Validation and next hardware run

CPU checks compare emitted patches against the authoritative full builder across 4000 generated mutation rounds, varying query count and graph shape. They compare all four materialized outputs and fused indices, and cover rollback, restore, cache ownership and MTP reuse. Existing backend-op cases, mirrored-view copies and all 14 speculative-MTP groups pass. These checks establish no CUDA throughput result.

On a funded two-card box, run the registered strict direct-attention cases for each selected physical card. They cover Q8/Q4 caches, 400128 physical cells, query counts one/two/three, changed indices and empty rows. Each card must produce the full 16/16 receipt; no CUDA device or unsupported operation is a failure. These per-card tests are followed by tensor-parallel component and real-pack checks; they do not substitute for them.

Record the real-pack baseline and candidate under the same six-slot, 1048576-cell pool, batch 4096, microbatch 512, Q8 K/V/indexer cache and draft-MTP depth-two configuration. Record actual occupied prompt tokens for every request. Include a single session at 400K and two simultaneous 400K sessions, with subsequent cache-kept turns. Preserve the exact engine SHA, build, pack, selected devices, flags and environment in each result. The older 512K/three-slot results cannot stand in for this configuration.

Report the server's per-request decode rate directly, using `(predicted_n - 1) / predicted_seconds` only if that rate is absent. For concurrent output throughput, divide all delivered output tokens by their common elapsed request interval; do not sum per-request rates. Missing or invalid timing evidence fails qualification. Keep actual MTP verification/replay counters and acceptance distributions with the rate, and record CUDA errors and output comparisons before promoting the engine pin.
