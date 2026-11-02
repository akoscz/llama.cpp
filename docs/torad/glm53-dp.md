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
build-dp/bin/test-backend-meta-allreduce-copy --direct 4
GGML_CUDA_AR_BF16_THRESHOLD=0 GGML_CUDA_ALLREDUCE=internal \
  build-dp/bin/test-llama-archs --glm-dp-device CUDA0,CUDA1,CUDA2,CUDA3 -s 1234
GGML_CUDA_AR_BF16_THRESHOLD=0 GGML_CUDA_ALLREDUCE=internal \
  build-dp/bin/test-speculative-mtp --dp-device CUDA0,CUDA1,CUDA2,CUDA3
```

The MTP test compares every draft and greedy verification result with an unsplit model, and exercises forced acceptance of zero, one and two drafts after slot reuse. The untrained fixture has no naturally accepted greedy drafts; these are correctness checks, not an acceptance-rate benchmark. Run it under Compute Sanitizer as well. The backend handles failed CUDA graph updates by re-instantiating the graph; use `--report-api-errors no` to check device memory without counting that handled update result as a memory failure.

- Synthetic GLM tests cover two/four ranks, mixed streams, microbatch boundaries, cross-owner forks, per-sequence/full checkpoints, MTP hidden-store/head behavior and rollback against an unsplit model. CPU tests also check rollback against the kept-prefix oracle.
- CUDA dense kernels change arithmetic with row count. Single-device 16-row and 64-row controls themselves change one near-tied top token in the synthetic fixture. DP comparisons retain the KL limit of 1e-6 against both controls and require top-token agreement with one entire unsplit control. This is not a claim of byte-identical logits across batch shapes.
- Before promotion, run the actual model at 2K, 4K and 16K, including multi-turn coding/tool workloads and MTP draft/acceptance comparisons. Repeat with BF16 collective wires enabled.
- On four physical cards, check the per-device KV allocation is one quarter of the total, then exercise 40 concurrent approximately 30K prompts. Record peak VRAM during load, prefill, decode and graph capture; every card must remain under the requested 93 GB ceiling.
- Measure 1/12/40 decoding streams and one/four 45K prefills. Report aggregate throughput, per-stream throughput and inter-token latency while another request prefills. Use ABBA against two-card tensor mode on two of the same cards.
- The fallback candidate is 36 slots at `-ub 1024` (`-c 15925248`). It needs its own measurement. Smaller microbatches can improve responsiveness but do not guarantee the highest prefill rate.

The requested 40-slot Rig profile must use the measured peak plus 1 GiB as `min_vram_mib`. Until that exists, no measured VRAM threshold or production profile is claimed.
