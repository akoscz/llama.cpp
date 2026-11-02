# GLM-5.3-Flash with data-parallel attention

Experimental work in [PR #110](https://github.com/torad-labs/llama.cpp/pull/110), stacked on #92. Four RTX PRO 6000 96 GB cards and 40 long-context slots are the target. Memory capacity and speed on that hardware have not been measured. Do not treat the bandwidth estimates as benchmark results.

## What `-sm dp` does

Each sequence belongs to device `seq_id % device_count`. Its attention cache, recurrent state, k-pool and MTP hidden-state storage stay on that device. Dense weights are replicated. Routed experts are divided by expert index; a gather before each expert block and a reduction after it combine their work. A single stream can be slower than tensor parallelism because its dense path runs on one card.

The first implementation requires GLM5NEXT, equal device shares, all layers on GPU, flash attention and non-unified KV. CPU layer offload, unequal shares, unified KV and a hybrid tensor/data split are unsupported. Use `-fit off` and specify capacity explicitly.

## Build

Use the PR branch, a CUDA toolkit supporting the target cards, and the same build on every comparison:

```sh
git fetch origin train/glm53-dp
git switch train/glm53-dp
cmake -S . -B build-dp -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-dp --parallel 6 --target llama-server llama-perplexity test-llama-archs test-speculative-mtp test-backend-meta-split test-backend-meta-allreduce-copy
build-dp/bin/llama-server --list-devices
```

Record `git rev-parse HEAD`, the GGUF hashes, driver/toolkit versions and `nvidia-smi topo -m` with each result. `GGML_CUDA_ALLREDUCE=internal` selects the fork's collective path. Check the initialization log for the actual path; the host-staged fallback is functional but does not establish PCIe peer-to-peer performance.

## Candidate serving command

Start with four slots for qualification. Increase `GLM_SLOTS` only after checking loaded and active memory peaks. Forty slots give `-c 17694720`; each slot has 442,368 cells. Advertise 409,600 tokens in the client/provider configuration and reserve 32,768 for the reply.

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export GGML_CUDA_ALLREDUCE=internal
export GGML_CUDA_GRAPH_MAX=0
export LLAMA_MTP_DEVICE_H=1
export LLAMA_TOK_EMBD_GPU=1

# Start quality comparisons with F32 collective wires.
export GGML_CUDA_AR_BF16_THRESHOLD=0
# Optional qualification counter; prints per-layer/per-device expert hits at clean shutdown.
export GGML_CUDA_DP_EXPERT_STATS=1

GLM_MODEL=/path/to/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00005.gguf
GLM_SLOTS=4
GLM_CTX=$((GLM_SLOTS * 442368))

build-dp/bin/llama-server -m "$GLM_MODEL" \
  -ngl all -fit off -sm dp -ts 1,1,1,1 \
  -np "$GLM_SLOTS" -c "$GLM_CTX" -ub 512 -b 2048 --prompt-step 512 \
  -ctk q8_0 -ctv q8_0 -ctki q8_0 -cts q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 -ubd 512 -ctkd q8_0 -ctvd q8_0 \
  -fa on --jinja --chat-template-file models/templates/GLM-5.3-Flash.jinja \
  --metrics --host 127.0.0.1 --port 8080
```

Set `GLM_MODEL` to the actual first shard; the filename above is a placeholder. Keep the remaining shards beside it. The pack must contain the MTP block. Add the pack's matching `--mmproj` when serving image inputs. Do not add `--kv-unified`.

Draft Q8 and recurrent Q8 are capacity candidates, not a measured quality recommendation. Qualify them against F16 draft cache and F32 recurrent state; the existing two-card Rig recipe uses F16 draft cache after a prior Q8 reply divergence. Changing either format changes the memory budget. Preserve the same template, tool definitions, message history and sampling settings in A/B runs.

After the F32-wire run passes, unset `GGML_CUDA_AR_BF16_THRESHOLD` and repeat quality and performance measurements. Disable expert counters for timing runs. Counters accumulate one hit per selected routed expert at the up projection, including graph replay; `first` identifies the first expert on that device, and `counts` follows expert-index order. Counts include warmup and prompt/decode work until clean shutdown.

## Qualification and capacity

Local development used an RTX 5080 and RTX 5070 Ti without peer-to-peer access. Four logical ranks on those two cards validate partition mechanics; they do not validate four-card topology, capacity or speed.

On the four-card host, run the component and synthetic-model checks before loading the full pack:

```sh
build-dp/bin/test-backend-meta-allreduce-copy --direct 4 --require-p2p
GGML_CUDA_AR_BF16_THRESHOLD=0 GGML_CUDA_ALLREDUCE=internal \
  build-dp/bin/test-llama-archs -a glm5next -s 1234
GGML_CUDA_AR_BF16_THRESHOLD=0 GGML_CUDA_ALLREDUCE=internal \
  build-dp/bin/test-llama-archs --glm-dp-device CUDA0,CUDA1,CUDA2,CUDA3 -s 1234
GGML_CUDA_AR_BF16_THRESHOLD=0 GGML_CUDA_ALLREDUCE=internal \
  build-dp/bin/test-speculative-mtp --dp-device CUDA0,CUDA1,CUDA2,CUDA3
```

The direct collective test covers host staging and peer paths on F32/BF16 wires, including tails and transfers larger than 32 MiB. `--require-p2p` fails if a requested peer path falls back to host memory. Without it, a fallback is labeled in the output. For three/four ranks it also runs an actual two-rank reference with the same two contributions and zero-valued extra ranks; this is bit-identical. Arbitrarily regrouping nonzero floating-point contributions is not guaranteed to preserve bits.

The standard `-a glm5next` command includes DP 2 and DP 4 with default and Q8 caches. With fewer than four CUDA devices, repeated devices are labeled logical ranks; a CPU-only build uses logical CPU ranks. The full CPU command passes locally. On the local CUDA cards the DP configurations pass, then the existing unsplit generated-prefix rollback comparison fails; the executable from before this test integration reproduces that failure. The focused `--glm-dp-device` command isolates DP results but does not waive the unsplit failure.

The MTP test compares every draft and greedy verification result with an unsplit model, and exercises forced acceptance of zero, one and two drafts after slot reuse. The untrained fixture has no naturally accepted greedy drafts; these are correctness checks, not an acceptance-rate benchmark. Run it under Compute Sanitizer as well. The backend handles failed CUDA graph updates by re-instantiating the graph; use `--report-api-errors no` to check device memory without counting that handled update result as a memory failure.

- Synthetic GLM tests cover two/four ranks, mixed streams, microbatch boundaries, cross-owner forks, per-sequence/full checkpoints, MTP hidden-store/head behavior and rollback against an unsplit model. CPU tests also check rollback against the kept-prefix oracle.
- CUDA dense kernels change arithmetic with row count. Single-device 16-row and 64-row controls themselves change one near-tied top token in the synthetic fixture. DP comparisons retain the KL limit of 1e-6 against both controls and require top-token agreement with one entire unsplit control. This is not a claim of byte-identical logits across batch shapes.
- Before promotion, run the actual model at 2K, 4K and 16K, including multi-turn coding/tool workloads and MTP draft/acceptance comparisons. Repeat with BF16 collective wires enabled.
- On four physical cards, check the per-device KV allocation is one quarter of the total, then exercise 40 concurrent approximately 30K prompts. Record peak VRAM during load, prefill, decode and graph capture; every card must remain under the requested 93 GB ceiling.
- Measure 1/12/40 decoding streams and one/four 45K prefills. Report aggregate throughput, per-stream throughput and inter-token latency while another request prefills. Use ABBA against two-card tensor mode on two of the same cards.
- The fallback candidate is 36 slots at `-ub 1024` (`-c 15925248`). It needs its own measurement. Smaller microbatches can improve responsiveness but do not guarantee the highest prefill rate.

The requested 40-slot Rig profile must use the measured peak plus 1 GiB as `min_vram_mib`. Its `draft_cache = { k = "q8_0", v = "q8_0" }` overrides the draft formats only for that profile. The draft microbatch inherits the target's 512 rows when `-ubd` is absent. Until the hardware measurements exist, no measured VRAM threshold or production profile is claimed.

## Four-card measurements (Nov 2, box 58861766)

Four RTX PRO 6000 Blackwell Server Edition (97,887 MiB each by nvidia-smi, all four on one NUMA node over PCIe, no NVLink), EPYC 9554, driver 580.105.08, CUDA 13.0. The fork at 9018dec92 (train/glm53-dp), built with `-DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF`. The pack measured is the one rig serves and pins by hash, BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF at 1bc0afad, IQ3_XXS in three shards (sha256 verified against `heads/glm-5.3-flash/head.toml`): the orcarouter abliteration of zai-org/GLM-5.3-Flash, quantized by BoldingBuilds. The Unsloth pack above was not measured. The runbook that drove the box is `scripts/glm53/dp-qualification.sh`'s ancestor kept with the session; every number below is in its logs.

Component checks on the four cards: `test-backend-meta-allreduce-copy --direct 4 --require-p2p` 66 ok (peer copy, host linked-list, host chunked; F32 and BF16 wires up to 16,777,219 elements); `test-llama-archs --glm-dp-device CUDA0,CUDA1,CUDA2,CUDA3 -s 1234` exit 0, "GLM dp 4: mean KL 0, 0 differing top tokens"; `test-speculative-mtp --dp-device CUDA0,CUDA1,CUDA2,CUDA3` exit 0; `test-llama-archs -a glm5next -s 1234` fails on the unsplit generated-prefix comparison recorded above as pre-existing (max_abs 2.96e-06), after its DP configurations pass.

### Two crashes at load, fixed in 9018dec92

The 40-slot server of the candidate command did not load on 9bd1f9c61. Each crash has its test in `tests/test-llama-archs.cpp`, written first and failing before the fix, and both reproduce on a CPU build (the DP ranks run on the CPU device there):

- `glm5next.cpp:420: GGML_ASSERT(n_tokens == n_tps*n_stream)` in the slot-graph reserve (`llama-context.cpp`): the reserve measured 32 rows over min(n_seqs, 32) sequences while the full memory context views every stream of the cache, so with more than 32 slots the indexer scoring found 32 rows over 40 streams. The reserve now keeps every sequence (test_glm_many_slots). Any server over 32 slots crashed this way, DP or not.
- `ggml-backend-meta.cpp:2835: GGML_ASSERT(size % chunk_size_full == 0)` on the server's two-token probe at load (common_context_can_seq_rm): the MTP head graph set `t_h_nextn` to every row and the masked read took the first n_outputs of them, a partial read of a split tensor under DP, and the wrong rows on one device whenever a step has fewer outputs than rows. The head now hands the masked read the output rows (test_glm_nextn_masked_rows).

### Memory at 40 and 36 slots

Per card at 40 slots (`-c 17694720`, 442,368 cells a slot, q8_0 caches, ub 512): model 39,218 MiB, attention cache 25,245, indexer cache 18,934, recurrent state 1,371, the MTP head's caches 2,295 and 1,721, compute 2,206 plus the head's 1,529, three graph slots up to 278 each. Load peak 93,861 MiB on the busiest card, 95,685 MiB resident after load: 2.2 GB free. That is not enough: the graphs grow past the reserve at depth under DP, and the 12-stream run below ended the server with `cudaMalloc failed: out of memory` allocating 603.83 MiB on device 0 in `ggml_backend_sched_alloc_graph` (process_ubatch). The 40-slot profile the PR asks for cannot be served on the 96 GB class.

At 36 slots (`-c 15925248`): load peak 90,459 MiB, 90,473 resident after load, 90,819 after a one-stream 400k session, 94,649 during twelve concurrent 400k prefills at ub 512 (the runtime growth is about 4 GB a card under that load). At ub 2048 the same server loads at 95,459 MiB and reaches 96,143 under the twelve prefills.

### 400k sessions, Claude Code's shape

`scripts/glm53/dp_longctx.py` builds a history the way Claude Code sends one: a system prompt, twelve tool definitions, a user request, then Read calls and their results from a public corpus (the fork's `src/`) until the server's own count (`/apply-template` and `/tokenize`) reaches the target; each later turn appends the reply, a question and one more Read, so the prefix stays byte for byte and the cache can hold. Sessions here measured 395k to 400k tokens (the builder's first version let two sessions through at 443k and 458k, over the slot; the server answered 400 and the builder was fixed in 9018dec92: the count decides, never the estimate).

One stream alone (36 slots, ub 512): a 395,395-token history read in 235.2 s, 1,681 tokens a second on its one card; the reply decoded at 37.9 tokens a second; turn 2 found the whole history in the cache (646 new tokens over 395,391 cached, 0.88 s) and decoded at 42.1; turn 3 at 46.9. No cache lost. (At 40 slots, before the memory ran out: 419,337 tokens in 252.1 s, 1,664 a second, decode 34.9 / 55.3 / 37.9.)

Twelve streams at once, each a 400k history from empty: the per-stream prefill rate is 100 to 136 tokens a second whatever the step or the microbatch, measured at ub 512 with the 512 step (130 a second a stream, 1,560 aggregate), at ub 512 with the 2048 step (115 to 136), and at ub 2048 with the 2048 step (101 to 103). Four cards together read twelve prompts no faster than one card reads one: each card's throughput falls from 1,681 tokens a second with one stream to about 300 to 390 with three. A stream decoding beside eleven such prefills produced 166 tokens in 84.0 s, 2.0 tokens a second, and its 646-token turn took 17.1 s to read. A 400k session opened while eleven others open theirs is ready after about an hour, against four minutes alone. The twelve-stream run was not taken to completion: turns 2 and 3 under concurrency are not measured.

The same history through `/v1/messages`, the way Claude Code sends it (system apart, tools as input_schema, Read calls as tool_use blocks, results as tool_result blocks), 392,251 input tokens on the 36-slot server: answered in 233.4 s with a tool_use block (a Read of `src/llama-quant.cpp`), stop_reason tool_use. The Anthropic path renders and serves a 400k agentic history.

### Why twelve streams read no faster than one

The branch's `-sm dp` is sequence parallel for attention and expert parallel for the MoE: each card keeps its own sequences' caches and the dense weights, and holds a quarter of the 288 routed experts. Every MoE layer (`src/models/glm5next.cpp`, `ggml_backend_meta_gather` before `build_moe_ffn`, `ggml_backend_meta_reduce` and `ggml_backend_meta_scatter` after it; the shared expert runs on each card's own rows) gathers the whole microbatch's rows onto every card, runs the card's 72 experts over all of them, and sums the four partials. Both collectives move the whole batch's rows, n_embd 4096 in F32, 16 KB a row: 8 MB a layer for a 512-row microbatch, 32 MB for 2048 rows, over 45 layers, and the four cards are on PCIe through the host, no NVLink.

The two collectives take different paths in `ggml/src/ggml-cuda/allreduce.cu`. The reduction over 1 MB goes by copy engine card to card (`ggml_cuda_ar_pipeline_init`: "copy-engine receives card to card, 96 MB receive scratch per GPU" in the server's log, the probe having found peer stores work on these cards): each card copies its 32 MB partial into three peers' receive buffers and adds what it received, 96 MB out and 96 MB in per card per layer, about 4.3 GB each way a 2048-row microbatch. The gather never takes the copy engines (`ggml_cuda_ar_allgather`: "the words go through the kernels whatever their size, so a capture holds the gather"): above the 128 KB peer-kernel limit it runs `ggml_cuda_ar_kernel_n` through pinned host memory in 1 MB chunks (`host_buf`, the "1024 KB chunked kernel staging"), each chunk a kernel store to the host, an arrival counter all four cards wait on, then every card's kernel reading the four chunks back from the host: 32 chunks a layer at 2048 rows, about 1,400 host round trips with a four-card barrier in each per microbatch, 360 at 512 rows. The per-microbatch time follows the rows: one stream (512 rows, 305 ms a microbatch, 1,681 tokens a second) and twelve streams (2,048 rows, about 1.7 s, 1,220 a second) cost the same per row, and the rows of twelve streams are no cheaper than the rows of one. With 8 of 288 experts a token, a token reaches a given card's experts with probability about 1 - (3/4)^8 = 0.90, so a top-k dispatch would carry nearly the same rows as the gather: the volume is in the model's shape, the path is the code's choice.

NCCL on the same tree (`-DGGML_CUDA_NCCL=ON` against the image's libnccl 2.28.3, GGML_CUDA_ALLREDUCE unset; its gather is an `ncclAllReduce` of the words, its reduction an `ncclAllReduce` in F32): twelve 400k streams at ub 512, step 2048, read at 119 to 129 tokens a second a stream after 90 s and 128 to 142 after seven minutes, against 115 to 136 on the internal collectives. Not the fix.

The server itself under nsys (`nsys profile --trace cuda,nvtx --cuda-graph-trace node`, twelve streams at 32k): a 66.5 s window of twelve concurrent prefills, then the server died under the profiler, `CUDA error: unspecified launch failure` 78 s in (the three twelve-stream runs without the profiler, five to seven minutes each, did not; what the profiler changes is not established). In the window each card is busy 50 to 52% of the time, and for 35% of it no card is busy at all, the host between launches. Of the kernel time summed over the four cards, 34.2% is the gather kernel, `ggml_cuda_ar_kernel_n<uint32>`: 39.7 s in 186,165 launches, median 89 µs, the longest 7.05 ms (a chunk waiting at its barrier for the slowest card). The two main expert matmul kernels (`mul_mat_q` IQ3_XXS at 16 and 80 rows) take 32.8 s together, the card-to-card copies of the reduction 11.5 s, host-to-device copies 6.6 s, the F32 reduction kernel for pieces under the copy-engine threshold 6.2 s. The cards spend more time in the gather than in the expert matmuls, and a third of the window idle.

This is the known shape of expert parallelism on PCIe hosts (ThunderEP, arXiv 2609.40093: "all inter-GPU transfers traverse CPU memory"; vLLM's EP guide requires NVLink or RDMA for its all-to-all backends and names the all-gather/reduce-scatter fallback as the only path without P2P), and the reason inference stacks serve such models on PCIe boxes as tensor parallel, where the per-layer all-reduce carries one row set instead of every card's. The branch has the pieces to do better than it does: the copy-engine peer path the reduction already takes, a reduce-scatter plus all-gather in place of four full partials (half the bytes), and a BF16 wire (`GGML_CUDA_AR_BF16_THRESHOLD`) for the rest.

### Two-card tensor mode on this tree

Rig's served profile (prefill-512k: `-sm tensor -ts 1,1`, three slots on one 524,288-cell unified pool, batch 2048/2048 step 1024, q8_0 caches, draft-mtp n_max 2 with the public vocab, GGML_CUDA_ALLREDUCE=internal) on cards 0 and 1, the same 395,395-token session on each build, main's first:

- main (28b26d62d, built the same way): the history read in 146.4 s, 2,700 tokens a second; the reply decoded at 51.8 tokens a second, turns 2 and 3 at 83.6 and 85.1 with the history in the cache. No cache lost. 87,155 MiB a card after the session.
- this tree (9018dec92): the read ran at 3,450 to 3,590 tokens a second until, 122,686 tokens in, the server died: `CUDA error: an illegal memory access was encountered` in `ggml_cuda_ar_acquire_slot` (`allreduce.cu:751`, `cudaEventSynchronize(p->ev_pool[i][slot].ker)`), at the same point on each of three runs (one at 419k, two at 395k).

So this tree breaks the two-card tensor profile rig serves, and main does not. The DP branch's 20 commits are the difference (the five PR #92 fixes are on both, as different SHAs). The same recipe at the branch's commits, one 250k session each, on the same cards (`llama-server` built at each commit, CUDA, arch 120):

- 974b2c611 (cuda: extend internal collectives to multiple peers): good, the 250k session read and three turns answered, cache kept.
- c875f8ff7 (llama: group ubatch rows by device and sequence): the 250k read completes (249,782 tokens), then turn 2, the cached prefix plus a short read, dies in `ggml-backend-meta.cpp:1357: GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis] == sum * tensor->ne[split_state.axis])`. 0d401ff8c (ggml: add explicit row layouts and meta gather/scatter), the commit before it, is good; 32eef41f3 (ggml: preserve ownership in slices of split cache buffers) is the first with this failure, and every later commit up to f78379aaa has it: `ggml_backend_meta_slice_view` now answers for every `GGML_OP_VIEW` of a split buffer in `ggml_backend_meta_get_split_state`, the tensor-split cache's views included.
- da8a76e5a (ggml: preserve device ownership through GLM ops and cache access), 9d77c323e, 7c07a29ac and the head: the illegal memory access at 122,686 tokens of turn 1, in `ggml_cuda_ar_acquire_slot`. f78379aaa (cuda: execute indirect matmuls over local expert ranges), the commit before da8a76e5a, reads the whole session and fails only at turn 2 as above, so the illegal access is da8a76e5a's.

Two regressions, then, both in tensor mode on two cards with no DP flag set: one in the second turn from the row grouping, one in the first read from the device-ownership work that followed it. Neither is covered by a test: `test-llama-archs` compares DP against batch-shape controls, and nothing on the branch runs the tensor recipe rig serves. The tests for both go in before the fixes, as the tests for the two load crashes above did.

### What this says

- The two load fixes and `dp_longctx.py` stand on their own: the slot-graph reserve bug hits any server over 32 slots, DP or not, and the session builder drives any llama-server.
- The 40-slot, 400k profile does not fit four 96 GB cards; 36 slots do, with 1 to 2 GB to spare under load.
- Alone, a 400k session is served well: 1,681 tokens a second to read, 38 to 47 to decode, the cache kept across turns. Beside eleven others it is not: a 400k read takes about an hour instead of four minutes, and decode beside the reads runs at 2 tokens a second. The rig profile this was for, many Claude Code sessions at 400k, is not served by this tree as it stands.
- The tree breaks the two-card tensor profile rig serves today, in two places. It cannot merge until both are fixed, with their tests.
- The path forward for DP prefill is in the gather: card to card by copy engine, as the reduction already goes, then reduce-scatter and a BF16 wire. Expert parallelism over PCIe will not read twelve prompts at four cards' speed, but it need not read them at one card's.
