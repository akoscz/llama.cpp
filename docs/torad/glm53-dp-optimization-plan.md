# DP4 throughput implementation proposal - November 4, 2026

Status: research and source inspection complete; the changes below are proposed, not implemented. No machine was started for this work. The existing `.Codex/PLAN.md` remains unchanged.

Start with owner-local sampling, then repair graph reuse where a cause is demonstrated, then optimize remaining collective latency. Keep the current BoldingBuilds IQ3_XXS pack, four-way DP with partitioned experts and replicated dense weights, private sequence caches, host latent KV, Q8 caches/state and MTP2. Preserve the two-card tensor serving path. Throughput comes before a 400K capacity run.

## Baseline and limits

Runtime: `82ad56cb8b67c83c0977ae9e4d23da9b5779d469`. Evidence: [November 4 profiling](glm53-dp.md#november-4-single-session-profiling-and-scaling-controls) and [measurements](glm53-dp-profile-20261104.json). Documentation head at inspection: `301547c2039b9321d9afab4371f3bdab1285ba94`, branch `fix/dp-sampler-reserve`. [PR 118](https://github.com/torad-labs/llama.cpp/pull/118) remains an open draft; its startup fix is the tested baseline, not yet main. Keep that fix separate from subsequent optimizations. Its PR description predates the completed profiling and should be refreshed when the PR is prepared for review.

One active session with 40 configured slots measured 45.3 tokens/s at 512 prompt tokens and 31.7 at 24,576. Reducing configured slots to one barely changed the short-context rate. One-slot host/GPU KV measured 36.7/36.8 at 24K with matching token IDs. These controls do not identify host KV or configured slot count as the main short-context bottleneck.

The traces show an extra full-vocabulary sampling gather and repeated graph instantiation. Collectives occupy 73-75% of summed GPU kernel durations, including rank waiting. This is neither wall time nor a removable percentage. The large unsigned gather family accounts for 21-22% of that sum, but not every captured invocation was attributed to a tensor; that percentage is not the predicted sampling speedup.

Concurrent 2/4/8-session replies already scale poorly. Their 64-token replies differ from the 128-token single-session controls. The measured common HTTP interval includes prefill and queueing; it is not aggregate steady decode throughput. MTP-off follows different output tokens and is also not an exact-work ablation.

## External results checked through Exa and primary sources

| Source | What was actually demonstrated | Application here |
| --- | --- | --- |
| [SGLang PR 5558](https://github.com/sgl-project/sglang/pull/5558), merged | DP attention's LM head can avoid gathering logits across DP groups by limiting vocabulary parallelism to the attention TP group. | Strong architectural precedent. Our DP owner already holds its full vocabulary row, so sampling need not mirror it across all four devices. |
| [llama.cpp PR 30662](https://github.com/ggml-org/llama.cpp/pull/30662), merged November 3 | Preserves the requested graph during idle eviction and gives cached graphs a reuse grace period after idle. | Already present in our `common.cuh:2079-2103`; do not repeat it as a new fix. |
| [Unsloth fork PR 232](https://github.com/unslothai/llama.cpp/pull/232), open | Author reports TP decode recovering when a graph-count cache smaller than its working set is enlarged; tested on V100/NVLink with another model. | Establishes a plausible failure mode, not our diagnosis. Our CUDA graph-count cap is already unlimited. Node limits, Meta cache replacement and new graph identities need separate attribution. |
| [SGLang graph design](https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph/) | Capture management separates execution phases, stable input buffers and padded shapes. | Use those invariants when fixing reuse. Copying a larger cache limit does not make incompatible shapes reusable. |
| [PCIe oneshot report](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/pcie-oneshot-allreduce.md) | Author measures small-collective improvements on four RTX PRO 6000s with Qwen/NVFP4, with results sensitive to topology and message size. | Our internal collective already uses direct exchange. Benchmark our SM-issued remote accesses and rank-arrival skew before selecting a different transport. A fast DMA copy does not establish fast kernel-issued peer access. Driver override claims are not independently qualified here. |
| [GLM on four RTX PRO 6000s](https://helix.ml/blog/glm53-flash-on-rtx-pro-6000-part-1) and [follow-up](https://helix.ml/blog/glm53-flash-tp2-rtx-pro-6000) | Same model family/hardware, different NVFP4 packs and SGLang layouts. The follow-up uses TP2 replicas with a shared 500K token pool per replica. | Useful feasibility evidence, not proof of forty simultaneous 400K sessions, our pack's performance or a replacement for the requested capacity architecture. |

Chinese forum searches included Linux.do, V2EX and Hostloc. The returned first-person reports include [four-card DeepSeek deployments](https://linux.do/t/topic/2754979?page=2), but do not reproduce our model, pack and DP/MTP layout. No exact-recipe solution was established by this search. [DeepEP's current requirements](https://github.com/deepseek-ai/DeepEP/blob/main/README.md#requirements) specify NVLink for intranode communication and RDMA for internode communication; it is not a ready-made PCIe-only replacement for our GGML path.

Search queries and fetched pages are retained locally under `.Codex/tmp/dp-research-20261104/`. Published speedups are not forecasts for this fork.

## 1. Remove the unnecessary sampling gather

Source evidence: `src/llama-graph.cpp:4773-4801` appends a dummy row before per-sequence sampling. `ggml/src/ggml-backend-meta.cpp:1784-1792` interprets padding along the split axis as a gather. DP logits are split by output row, so this gathers every vocabulary row onto every device. NVTX confirms this path executes on the full model.

Make the DP sampler consume the full vocabulary row on its owning device and preserve ownership through sampler outputs. Use explicit row ownership and an owner-local padding/view path; preserve the existing vocabulary-gather semantics for tensor split and the intentional MoE gathers. Return the selected token and only the outputs the caller actually requests from the owner. Keep the dummy/inactive-row mechanism's graph stability and alignment requirements.

Correctness constraints: preserve per-sequence RNG and penalty history, inactive samplers, empty ranks, uneven output rows, multiple MTP verification rows, candidates/probabilities and requested logits. Do not replace the real sampling chain with a greedy shortcut. A caller explicitly requesting full logits still receives them.

Extend existing `tests/test-backend-meta-sampler.cpp` and DP/MTP fixtures in `tests/test-speculative-mtp.cpp`. Add CPU-capable structural coverage that demonstrates the unwanted gather before the change, checks row ownership afterward, and retains tensor-split gather coverage. Same-device sampler comparisons must preserve greedy tokens and seeded chain results, including history, candidates and probabilities. Physical CUDA sampler equivalence remains a remote check; CPU routing success does not qualify CUDA arithmetic.

Done: no full-vocabulary cross-DP collective for owner-local sampling, passing ownership/reservation/MTP regressions, and an SM120a build ready for remote qualification. This removes demonstrated work; the speed gain is unmeasured.

## 2. Fix graph reuse using separate cache identities and eviction reasons

There are multiple independent layers:

- `src/llama-context.h:438-458`: the current graph and up to three retained graph/scheduler slots, with bounds on retained rows and buffers.
- `ggml/src/ggml-backend-meta.cpp:2592-2645`: four whole-evaluation capture entries keyed by graph UID, replaced by LRU. This bound matches the nominal context slots; its existence alone is not evidence it is too small.
- `ggml/src/ggml-cuda/common.cuh:2079-2162`: individual CUDA graphs, an unlimited count cap in this recipe, a 32,768-node default cap, and idle eviction with a grace period.

Add opt-in counters/trace events for graph UID, target versus draft phase, rows and per-rank ownership, reuse rejection reason, capture refusal, eviction reason, nodes retained, instantiate versus update versus replay. Record events at the existing decisions, without adding synchronization or `cudaMemGetInfo`. Preserve graph identity ownership and scheduler-buffer lifetime.

Use CPU fixtures to exercise repeated target/verify/draft shapes and identify unnecessary graph rebuilds where possible. Fix deterministic invalidation mistakes first. Resize a bounded cache only after establishing that live reusable entries exceed its capacity. If shapes or addresses change legitimately, correct the key/buffer lifetime or introduce bounded shape retention; do not replay a stale graph. Keep rank launches concurrent: the current per-device launcher prevents collective replay deadlocks.

Done: each warm repeated shape has an explainable replay/capture path; no repeated capture solely from a cache too small for the selected working set; retained graph/device memory is accounted for. A short remote event capture is allowed if CPU fixtures cannot identify the remaining cause. Do not disable all limits and call that a qualification.

## 3. Prepare the measurement harness before another rental

Extend `scripts/glm53/measure.py` and its existing tests. Use equal output lengths for 1/2/4/8 active sessions, fixed prompts/seeds, identical slot IDs and owner assignments, and retained token IDs. Report both native per-request timings and actual delivered-token counts over a common interval. Separate prefill/queueing from a window where all requested streams are decoding. MTP emits bursts, so preserve token counts and timestamps rather than assuming one SSE event equals one token. If streams have no common steady interval, report that explicitly.

Retain MTP draft/acceptance counters and target/draft service times. For code changes expected to preserve arithmetic, compare matching token trajectories; otherwise use a controlled token replay for attribution and report ordinary generation separately. Use repeated baseline/candidate order (ABBA) for the important cells. Instrumented runs do not supply the throughput baseline.

The MTP implementation already batches active sequences before each `llama_decode` (`common/speculative.cpp:2019-2062`). Do not propose adding batching that already exists. Investigate batching losses at scheduler boundaries, per-sequence output consumption, idle ranks and changes in accepted draft lengths only where evidence identifies them.

## 4. Optimize remaining collectives only after the first fixes

Extend the existing collective benchmark/test to measure actual decode payload sizes on four ranks, both simultaneous arrivals and controlled owner delays, with CUDA graphs on. Compare our LL/direct-peer and fallback paths, and measure kernel-issued peer accesses separately from copy-engine transfers. Include topology/NUMA placement and latency distributions. No model download is needed for this probe.

Choose the smallest supported change from those results: payload thresholds, unnecessary barriers, redundant gather data, or a transport change. Actual all-gather/reduce-scatter or routed dispatch/combine remains a later candidate if the all-reduce emulation is demonstrably the bottleneck. Preserve fixed reduction order, exact gather bits, empty ranks, BF16/F32 behavior, host fallback and two-device behavior. Do not adopt speculative driver overrides, new dependencies or expert replication without evidence and memory accounting.

MTP hidden-state transfers and expert-load imbalance remain secondary candidates. Attribute their cost before expanding this patch series. Keep MTP enabled for the serving candidate; the existing MTP-off short-context result does not justify removing it.

## Remote acceptance and stop conditions

Finish source changes, focused CPU regressions, CUDA compilation, review and packaging locally. Use one four-GPU machine for the next bounded qualification. Verify counter permissions before collecting any Nsight Compute data. Keep an independent provider stop timer and collect only the artifacts needed to decide the next step.

First qualify CUDA correctness, then warmed unprofiled 512/4K requests at 1/2/4/8 active sessions with 40 slots configured. Compare baseline and individual candidate changes as well as the combined candidate. Check that the sampling gather is absent and repeated graph instantiations fall for shapes already warmed. Extend the existing 24K control only after short-context throughput and scaling improve. Stop the machine as soon as collection finishes; analysis and documentation are local.

Do not launch a 400K/full-capacity fill while the short concurrent workload remains unusable. The eventual requirement is useful per-stream throughput at the accepted active-session count, followed by gradual 400K capacity qualification, tool-cache continuity and numerical checks. Forty allocated slots are not proof that forty filled sessions perform acceptably. The user's 70-80 tokens/s team target remains a target, not a predicted result from these fixes.
