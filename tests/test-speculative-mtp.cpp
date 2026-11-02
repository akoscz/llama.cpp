// The MTP drafter driven the way the server drives it: target verification batches, process, draft, accept, begin and
// end, on a small GLM-5.3 model with its NextN head.
//
// Goal: with --spec-draft-mtp-adaptive COST each sequence drafts either one token or --spec-draft-n-max tokens a round,
// from the acceptance of its own MTP drafts in the current request. It drafts the full depth while the extra tokens the
// positions past the first are expected to land, relative to a one-token round, exceed (n_max - 1) * COST. Shapes stay
// stable: a sequence's depth holds for windows of 8 of its rounds, the first window of a request is full depth, and a
// one-token window ends in one full-depth probe. Another sequence's rounds, an earlier request on the same slot, and a
// round verified twice (a checkpoint replay) never move it. Without the flag every round drafts n_max, as before.
// With --spec-draft-n-min m the low depth is max(1, m), capped at n_max, before the request's output limit shortens a draft.
// Full depth is kept while the tokens past the low depth pay for their added time; at m == n_max the depth stays n_max, and m > n_max disables drafting.
// When all drafts start landing after two windows capped at m, full depth returns within three windows and stays there.
// The head's kept hidden row stays the target row that a verification accepted, however the batch is laid out.
// A request with an image keeps drafting: the head takes the image's rows, so its cells stay consecutive with the target's.
// With LLAMA_MTP_DEVICE_H=1 the head reads a long target batch's hidden rows where the target left them, in device memory:
// the same drafts, kept rows and target logits as the host path, byte for byte, and only the kept rows reach the host.

#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"
#include "gguf.h"
#include "ggml-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int N_SEQ  = 3;
static constexpr int N_MAX  = 2;
static constexpr int WINDOW = 8;

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    seed ^= std::hash<std::string>()(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (auto & v : tmp) v = dis(gen);
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (auto & v : tmp) v = ggml_fp32_to_fp16(dis(gen));
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

// the GLM-5.3 fixture of test-llama-archs: KDA and DSA layers, routed experts, and one NextN block
static gguf_context_ptr glm_gguf() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_GLM5NEXT, ret.get());
    const uint32_t n_embd = 128, n_head = 4, n_ff = 192, n_layer = 3;
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(LLM_ARCH_GLM5NEXT));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                uint32_t(128));
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            uint32_t(128));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,      uint32_t(1));
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,     false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,               1.0f);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,   uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   std::vector<uint32_t>({1, 0, 1}));
    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,  8.0f);
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(0));
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-6f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         uint32_t(16));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,     uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,     uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,          uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,     uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,   uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,          uint32_t(4));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,          std::vector<uint32_t>({8, 8, 8, 8}));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                 1.0f);
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                  true);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,                      "no_vocab");
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        n_ff);
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);
    ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,              n_ff);
    ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,         uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_COUNT,                      uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,                 uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,               uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,                uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,                1.0f);
    ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,                 uint32_t(1));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,       2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,      uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,       uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,   n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,      uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,         uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,        true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND, -5.0f);
    return ret;
}

static llama_model_ptr glm_model(size_t seed, std::vector<ggml_backend_dev_t> devices = {},
        llama_split_mode split = LLAMA_SPLIT_MODE_LAYER) {
    auto gguf = glm_gguf();
    if (!devices.empty()) {
        llama_model_saver ms(LLM_ARCH_GLM5NEXT, gguf.get());
        ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(8));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT, uint32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(128));
    }
    llama_model_params mparams = llama_model_default_params();
    mparams.progress_callback = [](float, void *) { return true; };
    mparams.load_mtp = true;
    devices.push_back(nullptr);
    mparams.devices = devices.data();
    mparams.split_mode = split;
    llama_model_ptr model(llama_model_init_from_user(gguf.get(), set_tensor_data, &seed, mparams));
    if (!model) {
        throw std::runtime_error("failed to create the GLM model");
    }
    return model;
}

// one server: the target context, the MTP head's context made by the server's own path, and the drafter
struct server_rig {
    common_params params;
    llama_context_ptr tgt;
    common_speculative_init_result_ptr init;
    common_speculative_ptr spec;
    int32_t n_vocab = 0;
    int32_t n_max   = N_MAX;

    struct seq {
        llama_pos    n_past  = 0;
        llama_token  id_last = 0;
        llama_tokens prompt;
        llama_tokens draft;
    };
    std::vector<seq> seqs = std::vector<seq>(N_SEQ);

    // the target's logits of the last batch's output rows, in batch order
    std::vector<float> logits;

    server_rig(llama_model * model, float adaptive, int32_t n_max = N_MAX, int32_t n_min = 0, int32_t n_ubatch_dft = 0,
               ggml_type type_idx = GGML_TYPE_F16, int32_t n_ubatch_tgt = 512,
               bool private_streams = false, bool q8_cache = false) : n_max(n_max) {
        params.n_ctx      = 8192;
        params.cache_type_idx = type_idx;
        params.n_parallel = N_SEQ;
        params.n_batch    = 512;
        params.n_ubatch   = n_ubatch_tgt;
        params.kv_unified = !private_streams;
        if (private_streams) {
            params.n_ctx = N_SEQ*1024;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        params.cpuparams.n_threads       = 4;
        params.cpuparams_batch.n_threads = 4;
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = n_max;
        params.speculative.draft.n_min = n_min;
        params.speculative.draft.n_ubatch = n_ubatch_dft;
        params.speculative.draft.mtp_adaptive = adaptive;
        if (q8_cache) {
            params.cache_type_k = params.cache_type_v = params.cache_type_s = params.cache_type_idx = GGML_TYPE_Q8_0;
            params.speculative.draft.cache_type_k = params.speculative.draft.cache_type_v = GGML_TYPE_Q8_0;
        }
        const auto limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
        params.n_outputs_max         = limits.total;
        params.n_outputs_max_per_seq = limits.per_seq;

        tgt.reset(llama_init_from_model(model, common_context_params_to_llama(params)));
        if (!tgt) {
            throw std::runtime_error("failed to create the target context");
        }
        common_params params_dft = common_base_params_to_speculative(params);
        init = common_speculative_init_from_params(params_dft, model, tgt.get());
        if (!init->context()) {
            throw std::runtime_error("failed to create the MTP context");
        }
        params.speculative.draft.ctx_tgt = tgt.get();
        params.speculative.draft.ctx_dft = init->context();
        spec.reset(common_speculative_init(params.speculative, N_SEQ));
        if (!spec) {
            throw std::runtime_error("failed to create the drafter");
        }
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    }

    // the target computes a batch of rows, each sequence's rows consecutive; the drafter sees it as the server shows it
    // returns, per sequence, the batch index of its first row
    std::map<llama_seq_id, int> decode(const std::vector<std::pair<llama_seq_id, llama_tokens>> & rows) {
        int n = 0;
        for (const auto & r : rows) n += (int) r.second.size();
        llama_batch batch = llama_batch_init(n, 0, 1);
        std::map<llama_seq_id, int> first;
        for (const auto & r : rows) {
            // as the server asks: every row of a verification, the last row of a prompt
            const size_t n_out = r.second.size() <= (size_t) (1 + n_max) ? r.second.size() : 1;
            first[r.first] = batch.n_tokens;
            for (size_t i = 0; i < r.second.size(); ++i) {
                common_batch_add(batch, r.second[i], seqs[r.first].n_past++, { r.first }, i + n_out >= r.second.size());
            }
        }
        const int rc = llama_decode(tgt.get(), batch);
        const bool ok = rc == 0 && common_speculative_process(spec.get(), batch);
        logits.clear();
        for (int i = 0; ok && i < batch.n_tokens; ++i) {
            if (batch.logits[i]) {
                const float * row = llama_get_logits_ith(tgt.get(), i);
                logits.insert(logits.end(), row, row + n_vocab);
            }
        }
        llama_batch_free(batch);
        if (!ok) {
            throw std::runtime_error("failed to decode a target batch");
        }
        return first;
    }

    // an image piece of a sequence: n rows of embeddings and no tokens, shown to the drafter as the server's callback does
    void image(llama_seq_id s, const std::vector<float> & embd, int n) {
        const int n_embd_inp = llama_model_n_embd_inp(llama_get_model(tgt.get()));
        llama_batch batch = llama_batch_init(n, n_embd_inp, 1);
        std::memcpy(batch.embd, embd.data(), (size_t) n*n_embd_inp*sizeof(float));
        for (int i = 0; i < n; ++i) {
            batch.pos[i]       = seqs[s].n_past++;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = s;
            batch.logits[i]    = false;
        }
        batch.n_tokens = n;
        const bool ok = llama_decode(tgt.get(), batch) == 0 && common_speculative_process(spec.get(), batch);
        llama_batch_free(batch);
        logits.clear();
        if (!ok) {
            throw std::runtime_error("failed to decode an image piece");
        }
        seqs[s].prompt.insert(seqs[s].prompt.end(), n, LLAMA_TOKEN_NULL);
    }

    llama_tokens tokens(std::mt19937 & rng, int n) const {
        std::uniform_int_distribution<int> dis(0, n_vocab - 1);
        llama_tokens out(n);
        for (auto & t : out) t = dis(rng);
        return out;
    }

    // a new request on a slot: the last one released and its cells cleared, the prompt, then generation begins
    void request(llama_seq_id s, std::mt19937 & rng, int n_prompt) {
        common_speculative_end(spec.get(), s);
        llama_memory_seq_rm(llama_get_memory(tgt.get()), s, -1, -1);
        llama_memory_seq_rm(llama_get_memory(init->context()), s, -1, -1);
        common_speculative_set_state(spec.get(), s, {});
        seqs[s].n_past = 0;
        seqs[s].prompt = tokens(rng, n_prompt);
        decode({{ s, seqs[s].prompt }});
        common_speculative_begin(spec.get(), s, seqs[s].prompt);
        seqs[s].id_last = tokens(rng, 1)[0];
    }

    // the sequences given draft in one call; returns each one's draft length
    std::map<llama_seq_id, int> draft(const std::vector<llama_seq_id> & which) {
        for (const llama_seq_id s : which) {
            seqs[s].draft.clear();
            common_speculative_get_draft_params(spec.get(), s) = {
                /* .drafting = */ true,
                /* .n_max    = */ 64,
                /* .n_past   = */ seqs[s].n_past,
                /* .id_last  = */ seqs[s].id_last,
                /* .prompt   = */ &seqs[s].prompt,
                /* .result   = */ &seqs[s].draft,
            };
        }
        common_speculative_draft(spec.get());
        std::map<llama_seq_id, int> len;
        for (const llama_seq_id s : which) {
            len[s] = (int) seqs[s].draft.size();
            // the server drops the head's draft cells before the verification writes them again
            llama_memory_seq_rm(llama_get_memory(init->context()), s, seqs[s].n_past, -1);
        }
        return len;
    }

    // the verification batch of the drafted sequences, in the given row order
    std::map<llama_seq_id, int> verify(const std::vector<llama_seq_id> & order) {
        std::vector<std::pair<llama_seq_id, llama_tokens>> rows;
        for (const llama_seq_id s : order) {
            llama_tokens r = { seqs[s].id_last };
            r.insert(r.end(), seqs[s].draft.begin(), seqs[s].draft.end());
            rows.emplace_back(s, r);
        }
        return decode(rows);
    }

    void accept(llama_seq_id s, int n, std::mt19937 & rng) {
        common_speculative_accept(spec.get(), s, (uint16_t) n);
        seqs[s].id_last = tokens(rng, 1)[0];
    }
};

// which draft positions land: the first with probability p1, the next ones (given the first) with probability p2; none
// at or past position cap
struct acceptance {
    float p1  = 1.0f;
    float p2  = 1.0f;
    int   cap = 1 << 30;
    int operator()(int len, std::mt19937 & rng) const {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        int n = 0;
        while (n < std::min(len, cap) && u(rng) < (n == 0 ? p1 : p2)) {
            ++n;
        }
        return n;
    }
};

static void log_errors(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static bool fail(const char * what) {
    fprintf(stderr, "FAIL: %s\n", what);
    return false;
}

// set or clear an environment variable the drafter reads when it is created
static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// a rig made with LLAMA_MTP_DEVICE_H=1: target steps of 64 rows, head steps of 32
static std::unique_ptr<server_rig> device_rig(llama_model * model) {
    set_env("LLAMA_MTP_DEVICE_H", "1");
    std::unique_ptr<server_rig> rig;
    try {
        rig = std::make_unique<server_rig>(model, 0.0f, N_MAX, 0, 32, GGML_TYPE_F16, 64);
    } catch (...) {
        set_env("LLAMA_MTP_DEVICE_H", nullptr);
        throw;
    }
    set_env("LLAMA_MTP_DEVICE_H", nullptr);
    return rig;
}

static std::string str(const std::vector<int> & v) {
    std::string s;
    for (const int x : v) s += std::to_string(x);
    return s;
}

// draft lengths from a window boundary, replayed rounds left out: every window full depth or 7 low-depth rounds and a
// full-depth probe, the last window a prefix of either; a request's first window is full depth
static bool stable_shape(const std::vector<int> & len, bool request_start = true, int n_max = N_MAX, int n_low = 1) {
    for (size_t w = 0; w*WINDOW < len.size(); ++w) {
        bool full = true, low = w > 0 || !request_start;
        for (size_t i = w*WINDOW; i < std::min(len.size(), (w + 1)*WINDOW); ++i) {
            full = full && len[i] == n_max;
            low  = low  && len[i] == (i % WINDOW == WINDOW - 1 ? n_max : n_low);
        }
        if (!full && !low) {
            return false;
        }
    }
    return true;
}

// --spec-draft-ubatch: the head reads each target batch in steps of at most N rows, so its compute is sized for N rows
// while the target keeps its own step, and every draft is the one a head reading the whole batch at once makes: prompts
// longer than a step, a batch carrying two sequences' prompts across a step boundary, then rounds of drafts, verifications
// and partial acceptance
static bool test_draft_ubatch(llama_model * model) {
    server_rig whole(model, 0.0f);
    server_rig steps(model, 0.0f, N_MAX, 0, 32);
    if (llama_n_ubatch(steps.init->context()) != 32) return fail("the head does not step at --spec-draft-ubatch's 32 rows");
    if (llama_n_ubatch(steps.tgt.get()) != 512 || llama_n_ubatch(whole.init->context()) != 512) {
        return fail("the target's step, or the head's without the flag, is not the target's 512");
    }
    std::mt19937 rng_whole(7), rng_steps(7);
    for (llama_seq_id s = 0; s < N_SEQ; ++s) {
        whole.request(s, rng_whole, 70 + 37*s);
        steps.request(s, rng_steps, 70 + 37*s);
    }
    // two sequences' new prompt rows in one target batch, 45 + 45 rows: a 32-row step ends inside each of them
    const auto more = whole.tokens(rng_whole, 45), other = whole.tokens(rng_whole, 45);
    steps.tokens(rng_steps, 90);
    whole.decode({{0, more}, {1, other}});
    steps.decode({{0, more}, {1, other}});
    for (llama_seq_id s : {0, 1}) {
        whole.seqs[s].prompt.insert(whole.seqs[s].prompt.end(), s == 0 ? more.begin() : other.begin(), s == 0 ? more.end() : other.end());
        steps.seqs[s].prompt = whole.seqs[s].prompt;
    }
    const acceptance acc[N_SEQ] = { {1.0f, 1.0f}, {1.0f, 0.0f}, {0.5f, 0.5f} };
    for (int round = 0; round < 2*WINDOW; ++round) {
        whole.draft({0, 1, 2});
        steps.draft({0, 1, 2});
        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            if (whole.seqs[s].draft != steps.seqs[s].draft) return fail("a head stepping 32 rows drafted other tokens");
        }
        whole.verify({1, 2, 0});
        steps.verify({1, 2, 0});
        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            const int n = acc[s]((int) whole.seqs[s].draft.size(), rng_whole);
            acc[s]((int) steps.seqs[s].draft.size(), rng_steps);
            whole.accept(s, n, rng_whole);
            steps.accept(s, n, rng_steps);
        }
    }
    return true;
}

// -ctki q8_0 holds the head's indexer cache in q8_0, as it does the target's: the load names both caches' type, and the
// head drafts its full depth every round
static bool test_draft_indexer_type(llama_model * model) {
    std::vector<std::string> lines;
    llama_log_set([](ggml_log_level level, const char * text, void * data) {
        log_errors(level, text, nullptr);
        if (strstr(text, "creating indexer KV cache")) static_cast<std::vector<std::string> *>(data)->push_back(text);
    }, &lines);
    std::unique_ptr<server_rig> rig;
    try {
        rig = std::make_unique<server_rig>(model, 0.0f, N_MAX, 0, 0, GGML_TYPE_Q8_0);
    } catch (...) {
        llama_log_set(log_errors, nullptr);
        throw;
    }
    llama_log_set(log_errors, nullptr);
    if (lines.size() != 2) return fail("the load did not create the target's and the head's indexer caches");
    for (const auto & line : lines) {
        if (line.find("type = q8_0") == std::string::npos) return fail("an indexer cache is not -ctki's q8_0");
    }
    std::mt19937 rng(3);
    for (llama_seq_id s = 0; s < N_SEQ; ++s) rig->request(s, rng, 40 + 11*s);
    for (int round = 0; round < 2*WINDOW; ++round) {
        const auto len = rig->draft({0, 1, 2});
        rig->verify({0, 1, 2});
        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            if (len.at(s) != N_MAX) return fail("a head on a q8_0 indexer cache drafted fewer than n_max tokens");
            rig->accept(s, 1, rng);
        }
    }
    return true;
}

// fixed depth: every round of every sequence drafts n_max whatever lands
static bool test_baseline(llama_model * model) {
    server_rig rig(model, 0.0f);
    std::mt19937 rng(1);
    for (llama_seq_id s = 0; s < N_SEQ; ++s) rig.request(s, rng, 9 + s);
    const acceptance acc[N_SEQ] = { {0.0f, 0.0f}, {1.0f, 0.0f}, {0.5f, 0.5f} };
    for (int round = 0; round < 4*WINDOW; ++round) {
        const auto len = rig.draft({0, 1, 2});
        rig.verify({2, 0, 1});
        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            if (len.at(s) != N_MAX) return fail("fixed depth drafted fewer than n_max tokens");
            rig.accept(s, acc[s](len.at(s), rng), rng);
        }
    }
    return true;
}

// one batch, three sequences: the one whose drafts all land keeps the full depth; the one whose drafts never land and
// the one whose second token never lands drop to one token after the first window, probing once a window. A new request
// on a slot starts at full depth again, and leaves the other slots as they were. A sequence that starts to land again
// returns to the full depth within three windows
static bool test_adaptive(llama_model * model) {
    server_rig rig(model, 0.18f);
    std::mt19937 rng(2);
    for (llama_seq_id s = 0; s < N_SEQ; ++s) rig.request(s, rng, 7 + 2*s);
    acceptance acc[N_SEQ] = { {1.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f} };
    std::vector<int> len[N_SEQ];
    auto run = [&](int rounds) {
        for (int round = 0; round < rounds; ++round) {
            const auto l = rig.draft({0, 1, 2});
            rig.verify(round % 2 ? std::vector<llama_seq_id>{1, 2, 0} : std::vector<llama_seq_id>{0, 1, 2});
            for (llama_seq_id s = 0; s < N_SEQ; ++s) {
                len[s].push_back(l.at(s));
                rig.accept(s, acc[s](l.at(s), rng), rng);
            }
        }
    };
    run(3*WINDOW);
    const std::string low = "11111112";
    const std::string full(WINDOW, '0' + N_MAX);
    if (str(len[0]) != full + full + full) return fail("a sequence whose drafts all land left the full depth");
    if (str(len[1]) != full + low + low) return fail("a sequence whose drafts never land did not drop to one token");
    if (str(len[2]) != full + low + low) return fail("a sequence whose second token never lands did not drop to one token");

    // a new request on slot 1: full depth from its first round; slots 0 and 2 keep their own depth
    rig.request(1, rng, 5);
    for (auto & l : len) l.clear();
    run(WINDOW);
    if (str(len[1]) != full) return fail("a new request inherited the depth of the slot's previous request");
    if (str(len[0]) != full) return fail("another slot's new request moved a full-depth sequence");
    if (str(len[2]) != low) return fail("another slot's new request moved a one-token sequence");

    // slot 2 starts to land everything: back to the full depth within three windows, and it stays there
    acc[2] = {1.0f, 1.0f};
    for (auto & l : len) l.clear();
    run(6*WINDOW);
    const std::string tail = str(len[2]).substr(3*WINDOW);
    if (tail != full + full + full) return fail("a sequence whose drafts land again did not return to the full depth");
    for (llama_seq_id s = 0; s < N_SEQ; ++s) {
        if (!stable_shape(len[s], false)) return fail("a sequence's depth changed inside a window");
    }
    return true;
}

// a round the target verifies twice (a checkpoint replay) is not an observation: a sequence whose every round is
// replayed keeps the full depth however little lands
static bool test_replay(llama_model * model) {
    server_rig rig(model, 0.18f);
    std::mt19937 rng(3);
    rig.request(0, rng, 11);
    for (int round = 0; round < 4*WINDOW; ++round) {
        const auto len = rig.draft({0});
        if (len.at(0) != N_MAX) return fail("a replayed sequence left the full depth");
        rig.verify({0});
        rig.seqs[0].draft.resize(1);
        rig.verify({0});
        rig.accept(0, 0, rng);
    }
    // the same rounds verified once each are observations: one window, then one token
    std::vector<int> len;
    for (int round = 0; round < 2*WINDOW; ++round) {
        len.push_back(rig.draft({0}).at(0));
        rig.verify({0});
        rig.accept(0, 0, rng);
    }
    if (str(len) != std::string(WINDOW, '0' + N_MAX) + "11111112") return fail("rounds verified once did not move the depth");
    return true;
}

struct history_round {
    std::vector<llama_seq_id> drafting; // the sequences that draft this round, in batch order
    std::vector<bool>         replay;   // per sequence of the round
    std::vector<acceptance>   acc;      // per sequence of the round
};

// generated rounds: acceptance moves in phases, so that depths change; some rounds leave sequences out or replay them
static std::vector<history_round> generate(std::mt19937 & rng, int n_rounds, const std::vector<llama_seq_id> & seqs) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<acceptance> phase(N_SEQ);
    std::vector<history_round> out;
    for (int round = 0; round < n_rounds; ++round) {
        if (round % 12 == 0) {
            for (auto & a : phase) {
                const float levels[] = { 0.0f, 0.2f, 0.6f, 1.0f };
                a = { levels[rng() % 4], levels[rng() % 4] };
            }
        }
        history_round r;
        for (const llama_seq_id s : seqs) {
            if (u(rng) < 0.85f) {
                r.drafting.push_back(s);
            }
        }
        std::shuffle(r.drafting.begin(), r.drafting.end(), rng);
        for (const llama_seq_id s : r.drafting) {
            r.replay.push_back(u(rng) < 0.1f);
            r.acc.push_back(phase[s]);
        }
        out.push_back(r);
    }
    return out;
}

// plays rounds; each sequence's acceptance draws from its own stream, so that a sequence's own rounds alone decide its
// draws. Returns, per sequence, its draft lengths with replayed rounds left out
static std::vector<std::vector<int>> play(server_rig & rig, const std::vector<history_round> & rounds, std::vector<std::mt19937> & draws) {
    std::vector<std::vector<int>> len(N_SEQ);
    std::mt19937 rng(7);
    for (const auto & r : rounds) {
        if (r.drafting.empty()) {
            continue;
        }
        const auto l = rig.draft(r.drafting);
        rig.verify(r.drafting);
        std::vector<llama_seq_id> replayed;
        for (size_t i = 0; i < r.drafting.size(); ++i) {
            if (r.replay[i]) {
                replayed.push_back(r.drafting[i]);
                rig.seqs[r.drafting[i]].draft.resize(draws[r.drafting[i]]() % (l.at(r.drafting[i]) + 1));
            }
        }
        if (!replayed.empty()) {
            rig.verify(replayed);
        }
        for (size_t i = 0; i < r.drafting.size(); ++i) {
            const llama_seq_id s = r.drafting[i];
            const int n_rows = r.replay[i] ? (int) rig.seqs[s].draft.size() : l.at(s);
            rig.accept(s, r.acc[i](n_rows, draws[s]), rng);
            if (!r.replay[i]) {
                len[s].push_back(l.at(s));
            }
        }
    }
    return len;
}

// over generated histories: shapes stay stable, no round drafts fewer than n_min tokens, a sequence's depths follow from
// its own rounds alone (the same rounds alone, or among other sequences' rounds, give the same lengths), and a slot's
// earlier request never reaches the next
static bool histories(llama_model * model, int n_max, int n_min, uint32_t n_seeds) {
    const int n_low = std::min(std::max(1, n_min), n_max);
    int n_low_rounds = 0;
    for (uint32_t seed = 0; seed < n_seeds; ++seed) {
        std::mt19937 rng(100 + seed);
        const auto rounds = generate(rng, 5*WINDOW, {0, 1, 2});

        // all three sequences together
        server_rig together(model, 0.18f, n_max, n_min);
        std::vector<std::mt19937> draws = { std::mt19937(seed*3 + 0), std::mt19937(seed*3 + 1), std::mt19937(seed*3 + 2) };
        for (llama_seq_id s = 0; s < N_SEQ; ++s) together.request(s, rng, 6 + s);
        const auto len = play(together, rounds, draws);

        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            if (n_low < n_max) {
                n_low_rounds += (int) std::count(len[s].begin(), len[s].end(), n_low);
            }
            if (std::any_of(len[s].begin(), len[s].end(), [&](int l) { return l < std::max(1, n_min); })) {
                fprintf(stderr, "n_max %d n_min %d seed %u seq %d: %s\n", n_max, n_min, seed, s, str(len[s]).c_str());
                return fail("a generated history drafted fewer than n_min tokens");
            }
            if (!stable_shape(len[s], true, n_max, n_low)) {
                fprintf(stderr, "n_max %d n_min %d seed %u seq %d: %s\n", n_max, n_min, seed, s, str(len[s]).c_str());
                return fail("a generated history changed a depth inside a window");
            }
            // the same sequence's rounds alone, after an earlier request of its own on the slot
            std::vector<history_round> own;
            for (const auto & r : rounds) {
                history_round o;
                for (size_t i = 0; i < r.drafting.size(); ++i) {
                    if (r.drafting[i] == s) {
                        o.drafting = { s };
                        o.replay   = { r.replay[i] };
                        o.acc      = { r.acc[i] };
                    }
                }
                own.push_back(o);
            }
            server_rig alone(model, 0.18f, n_max, n_min);
            std::vector<std::mt19937> earlier = { std::mt19937(99), std::mt19937(98), std::mt19937(97) };
            alone.request(s, rng, 5);
            std::vector<history_round> before = generate(rng, 3*WINDOW, {s});
            for (auto & r : before) {
                for (auto & a : r.acc) a = { 0.0f, 0.0f };
            }
            play(alone, before, earlier);
            alone.request(s, rng, 8);
            std::vector<std::mt19937> draws_alone = { std::mt19937(seed*3 + 0), std::mt19937(seed*3 + 1), std::mt19937(seed*3 + 2) };
            const auto len_alone = play(alone, own, draws_alone);
            if (len_alone[s] != len[s]) {
                fprintf(stderr, "n_max %d n_min %d seed %u seq %d: together %s, alone %s\n", n_max, n_min, seed, s, str(len[s]).c_str(), str(len_alone[s]).c_str());
                return fail("a sequence's depths depend on other sequences or on the slot's earlier request");
            }
        }
    }
    // the histories hold phases where nothing lands: where a low depth exists, some rounds must have drafted it
    if (n_low < n_max && n_low_rounds == 0) return fail("no generated history ever drafted the low depth");
    return true;
}

static bool test_histories(llama_model * model) {
    return histories(model, N_MAX, 0, 6);
}

// --spec-draft-n-min: three sequences in one batch, whose drafts all land, never land, or land up to n_min tokens and
// none past. The full-depth one keeps n_max. The other two drop to n_min after the first window, with one full-depth
// probe a window: the tokens past n_min never land, so they do not pay for their cost. No round drafts fewer than n_min
// tokens or skips verification. At n_min = n_max every round drafts n_max. Then the same over generated histories, for
// n_max up to 4
static bool test_n_min(llama_model * model) {
    struct config { int n_max, n_min; };
    for (const config c : { config{2, 2}, config{3, 2}, config{4, 2}, config{4, 3}, config{4, 4}, config{3, 5} }) {
        server_rig rig(model, 0.18f, c.n_max, c.n_min);
        std::mt19937 rng(20 + 8*c.n_max + c.n_min);
        for (llama_seq_id s = 0; s < N_SEQ; ++s) rig.request(s, rng, 7 + 2*s);
        const acceptance acc[N_SEQ] = { {1.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 1.0f, c.n_min} };
        std::vector<int> len[N_SEQ];
        for (int round = 0; round < 3*WINDOW; ++round) {
            const auto l = rig.draft({0, 1, 2});
            rig.verify(round % 2 ? std::vector<llama_seq_id>{2, 0, 1} : std::vector<llama_seq_id>{0, 1, 2});
            for (llama_seq_id s = 0; s < N_SEQ; ++s) {
                len[s].push_back(l.at(s));
                rig.accept(s, acc[s](l.at(s), rng), rng);
            }
        }
        // n_min past n_max: no draft is ever long enough, with the flag or without it
        const bool none = c.n_min > c.n_max;
        const std::string full = none ? std::string(WINDOW, '0') : std::string(WINDOW, '0' + c.n_max);
        const std::string low  = c.n_min >= c.n_max ? full : std::string(WINDOW - 1, '0' + c.n_min) + std::string(1, '0' + c.n_max);
        bool ok = str(len[0]) == full + full + full;
        ok = ok && str(len[1]) == full + low + low;
        ok = ok && str(len[2]) == full + low + low;
        if (!ok) {
            fprintf(stderr, "n_max %d n_min %d: %s %s %s\n", c.n_max, c.n_min, str(len[0]).c_str(), str(len[1]).c_str(), str(len[2]).c_str());
            return fail("with n_min, a sequence did not keep drafting at least n_min tokens at the depth its acceptance calls for");
        }
    }
    for (const config c : { config{2, 2}, config{3, 2}, config{4, 2}, config{4, 3} }) {
        if (!histories(model, c.n_max, c.n_min, 2)) return false;
    }
    for (const config c : { config{3, 2}, config{4, 2}, config{4, 3} }) {
        server_rig rig(model, 0.18f, c.n_max, c.n_min);
        std::mt19937 rng(50 + 8*c.n_max + c.n_min);
        rig.request(0, rng, 11);
        std::vector<int> len;
        for (int round = 0; round < 8*WINDOW; ++round) {
            const int n = rig.draft({0}).at(0);
            len.push_back(n);
            rig.verify({0});
            rig.accept(0, round < 2*WINDOW ? std::min(n, c.n_min) : n, rng);
        }
        const std::string full(WINDOW, '0' + c.n_max);
        const std::string low = std::string(WINDOW - 1, '0' + c.n_min) + std::string(1, '0' + c.n_max);
        if (str(len).substr(0, 2*WINDOW) != full + low) {
            return fail("a sequence capped at n_min did not reach low depth before acceptance improved");
        }
        if (!stable_shape(len, true, c.n_max, c.n_min)) {
            return fail("a sequence recovering from n_min changed depth inside a window");
        }
        if (str(len).substr(5*WINDOW) != full + full + full) {
            fprintf(stderr, "n_max %d n_min %d recovery: %s\n", c.n_max, c.n_min, str(len).c_str());
            return fail("a sequence whose drafts all land did not recover from n_min to full depth within three windows");
        }
    }
    return true;
}

// the kept row: after a batch, a sequence's state is its last row's position and the target's hidden row there; after
// accept(n) it is the verification's row n. Generated layouts: sequences in any order, prompt-length runs of rows
// longer than any draft, and verification-length runs; with the adaptive depth on and off
static bool test_kept_row(llama_model * model) {
    for (const float adaptive : {0.0f, 0.18f}) {
        server_rig rig(model, adaptive);
        std::mt19937 rng(adaptive > 0.0f ? 11 : 12);
        const int n_embd = llama_model_n_embd_out(llama_get_model(rig.tgt.get()));
        auto check = [&](llama_seq_id s, llama_pos pos, const float * h) {
            std::vector<uint8_t> state;
            if (!common_speculative_get_state(rig.spec.get(), s, state)) return false;
            if (state.size() != sizeof(llama_pos) + (size_t) n_embd*sizeof(float)) return false;
            llama_pos kept = -1;
            std::memcpy(&kept, state.data(), sizeof(llama_pos));
            return kept == pos && std::memcmp(state.data() + sizeof(llama_pos), h, (size_t) n_embd*sizeof(float)) == 0;
        };
        for (llama_seq_id s = 0; s < N_SEQ; ++s) rig.request(s, rng, 3 + s);
        // the head drafts once for every sequence, so that accept reaches it
        rig.draft({0, 1, 2});
        rig.verify({0, 1, 2});
        for (int step = 0; step < 40; ++step) {
            std::vector<llama_seq_id> order = {0, 1, 2};
            std::shuffle(order.begin(), order.end(), rng);
            order.resize(1 + rng() % N_SEQ);
            std::vector<std::pair<llama_seq_id, llama_tokens>> rows;
            for (const llama_seq_id s : order) {
                const int n = rng() % 3 == 0 ? 5 + rng() % 40 : 1 + rng() % (N_MAX + 1);
                rows.emplace_back(s, rig.tokens(rng, n));
            }
            std::map<llama_seq_id, llama_pos> pos0;
            for (const llama_seq_id s : order) pos0[s] = rig.seqs[s].n_past;
            const auto first = rig.decode(rows);
            std::map<llama_seq_id, std::vector<float>> h;
            for (const auto & r : rows) {
                const int n = (int) r.second.size();
                for (int i = 0; i < n; ++i) {
                    const float * row = llama_get_embeddings_nextn_ith(rig.tgt.get(), first.at(r.first) + i);
                    h[r.first].insert(h[r.first].end(), row, row + n_embd);
                }
                if (!check(r.first, pos0[r.first] + n - 1, h[r.first].data() + (size_t) (n - 1)*n_embd)) {
                    return fail("the kept row after a batch is not the sequence's last target row");
                }
            }
            for (const auto & r : rows) {
                if (rng() % 2) continue;
                const int n_rows = (int) r.second.size();
                const int n = rng() % (std::min(n_rows - 1, N_MAX + 1) + 1);
                common_speculative_accept(rig.spec.get(), r.first, (uint16_t) n);
                if (!check(r.first, pos0[r.first] + n, h[r.first].data() + (size_t) n*n_embd)) {
                    return fail("the kept row after accept is not the verification's accepted row");
                }
            }
        }
    }
    return true;
}

static double nmse(const float * a, const float * b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) a[i] * a[i];
    }
    return den > 0.0 ? num / den : num;
}

// a sequence's hidden rows do not depend on its batch neighbours: a batch carrying a longer prompt of one sequence before
// a shorter one of another (the hybrid memory's equal split interleaves them) gives every batch row the target row of
// its own sequence and position, as each prompt read alone does, and each sequence keeps its own last row. Generated:
// the two lengths and which sequence comes first
static bool test_rows_in_batch_order(llama_model * model) {
    const int n_embd = llama_model_n_embd_out(model);
    std::mt19937 gen(41);
    for (int trial = 0; trial < 6; ++trial) {
        const int n_long = 60 + (int) (gen() % 80), n_short = 5 + (int) (gen() % 50);
        const llama_seq_id s_first = trial % 2, s_second = 1 - s_first;
        server_rig alone(model, 0.0f), both(model, 0.0f);
        std::mt19937 rng_a(100 + trial), rng_b(100 + trial);
        for (llama_seq_id s = 0; s < 2; ++s) {
            alone.request(s, rng_a, 4 + s);
            both.request(s, rng_b, 4 + s);
        }
        const llama_tokens first = alone.tokens(rng_a, n_long), second = alone.tokens(rng_a, n_short);
        std::vector<float> h_alone[2];
        for (const auto & r : { std::make_pair(s_first, first), std::make_pair(s_second, second) }) {
            alone.decode({ r });
            for (int i = 0; i < (int) r.second.size(); ++i) {
                const float * row = llama_get_embeddings_nextn_ith(alone.tgt.get(), i);
                h_alone[r.first].insert(h_alone[r.first].end(), row, row + n_embd);
            }
        }
        const auto at = both.decode({{ s_first, first }, { s_second, second }});
        for (const llama_seq_id s : { s_first, s_second }) {
            const int n = s == s_first ? n_long : n_short;
            std::vector<float> h;
            for (int i = 0; i < n; ++i) {
                const float * row = llama_get_embeddings_nextn_ith(both.tgt.get(), at.at(s) + i);
                h.insert(h.end(), row, row + n_embd);
            }
            if (nmse(h_alone[s].data(), h.data(), h.size()) > 1e-6) {
                fprintf(stderr, "trial %d: %d rows then %d rows, seq %d: nmse %.3g\n", trial, n_long, n_short, s,
                        nmse(h_alone[s].data(), h.data(), h.size()));
                return fail("a batch row holds another row's hidden state when a longer prompt comes first");
            }
            std::vector<uint8_t> st_alone, st_both;
            if (!common_speculative_get_state(alone.spec.get(), s, st_alone) ||
                !common_speculative_get_state(both.spec.get(), s, st_both) || st_alone.size() != st_both.size() ||
                std::memcmp(st_alone.data(), st_both.data(), sizeof(llama_pos)) != 0 ||
                nmse((const float *) (st_alone.data() + sizeof(llama_pos)), (const float *) (st_both.data() + sizeof(llama_pos)),
                     (size_t) n_embd) > 1e-6) {
                return fail("a sequence's kept row is not its own last row when a longer prompt comes first");
            }
        }
    }
    return true;
}

// a request with an image: the target reads text, the image's rows (embeddings and no tokens, cut at n_batch as mtmd cuts
// a chunk, each piece shown to the drafter by the server's post-decode callback), then the text after it. The head takes
// every row, so its cells end where the target's end, the kept row is the last target row read, and the head drafts at
// full depth after the image. Generated: text and image lengths, images longer than n_batch, every slot in turn
static bool test_media(llama_model * model) {
    server_rig rig(model, 0.0f);
    std::mt19937 rng(31);
    const int n_embd     = llama_model_n_embd_out(model);
    const int n_embd_inp = llama_model_n_embd_inp(model);
    const int n_batch    = (int) rig.params.n_batch;
    auto * mem_tgt = llama_get_memory(rig.tgt.get());
    auto * mem_dft = llama_get_memory(rig.init->context());
    std::normal_distribution<float> embd_dis(0.0f, 1.0e-2f);
    auto kept = [&](llama_seq_id s, llama_pos pos, const std::vector<float> & h) {
        std::vector<uint8_t> state;
        if (!common_speculative_get_state(rig.spec.get(), s, state)) return false;
        if (state.size() != sizeof(llama_pos) + (size_t) n_embd*sizeof(float)) return false;
        llama_pos p = -1;
        std::memcpy(&p, state.data(), sizeof(llama_pos));
        return p == pos && std::memcmp(state.data() + sizeof(llama_pos), h.data(), (size_t) n_embd*sizeof(float)) == 0;
    };
    auto same_end = [&](llama_seq_id s) {
        return llama_memory_seq_pos_max(mem_dft, s) == llama_memory_seq_pos_max(mem_tgt, s) &&
               llama_memory_seq_pos_max(mem_tgt, s) == rig.seqs[s].n_past - 1;
    };
    for (int trial = 0; trial < 9; ++trial) {
        const llama_seq_id s = trial % N_SEQ;
        common_speculative_end(rig.spec.get(), s);
        llama_memory_seq_rm(mem_tgt, s, -1, -1);
        llama_memory_seq_rm(mem_dft, s, -1, -1);
        common_speculative_set_state(rig.spec.get(), s, {});
        rig.seqs[s].n_past = 0;
        const llama_tokens before = rig.tokens(rng, 1 + rng() % 8);
        rig.decode({{ s, before }});
        const int n_img = trial < N_SEQ ? n_batch + 1 + (int) (rng() % n_batch) : 1 + (int) (rng() % (2*n_batch));
        for (int i0 = 0; i0 < n_img; i0 += n_batch) {
            const int n = std::min(n_batch, n_img - i0);
            llama_batch batch = llama_batch_init(n, n_embd_inp, 1);
            for (int64_t i = 0; i < (int64_t) n*n_embd_inp; ++i) batch.embd[i] = embd_dis(rng);
            for (int i = 0; i < n; ++i) {
                batch.pos[i]       = rig.seqs[s].n_past++;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = s;
                batch.logits[i]    = false;
            }
            batch.n_tokens = n;
            bool ok = llama_decode(rig.tgt.get(), batch) == 0;
            std::vector<float> h_last;
            if (ok) {
                const float * row = llama_get_embeddings_nextn_ith(rig.tgt.get(), n - 1);
                h_last.assign(row, row + n_embd);
                ok = common_speculative_process(rig.spec.get(), batch);
            }
            llama_batch_free(batch);
            if (!ok) return fail("the drafter refused an image's rows");
            if (!kept(s, rig.seqs[s].n_past - 1, h_last)) return fail("the kept row after an image piece is not its last target row");
            if (!same_end(s)) return fail("the head's cells do not end where the target's do after an image piece");
        }
        const llama_tokens after = rig.tokens(rng, 1 + rng() % 8);
        try {
            rig.decode({{ s, after }});
        } catch (const std::exception &) {
            return fail("the text after an image did not reach the head");
        }
        if (!same_end(s)) return fail("the head's cells do not end where the target's do after the text that follows an image");
        llama_tokens prompt = before;
        prompt.insert(prompt.end(), n_img, LLAMA_TOKEN_NULL);
        prompt.insert(prompt.end(), after.begin(), after.end());
        rig.seqs[s].prompt = prompt;
        common_speculative_begin(rig.spec.get(), s, rig.seqs[s].prompt);
        rig.seqs[s].id_last = rig.tokens(rng, 1)[0];
        for (int round = 0; round < 3; ++round) {
            const auto len = rig.draft({ s });
            if (len.at(s) != N_MAX) return fail("the head drafted fewer than n_max tokens after an image");
            try {
                rig.verify({ s });
            } catch (const std::exception &) {
                return fail("a verification after an image did not reach the head");
            }
            rig.accept(s, (int) (rng() % (N_MAX + 1)), rng);
        }
    }
    return true;
}

// LLAMA_MTP_DEVICE_H=1: the head reads a long target batch's hidden rows where the target left them. On the served
// GLM-5.3 Flash each 2,048-row prefill step copied the target's 32 MiB of hidden rows from card 0 to the host, shifted
// them there and copied them to each card again for the head (the host-gap trace of Nov 1, 2026). With the switch, a
// target batch of 64 rows or more leaves every row in the store and the head gathers its rows there. Against a rig
// without it: after every batch the store holds that batch's rows exactly when it has 64 or more, the target's logits
// and every sequence's kept row are the same bytes, and every draft is the same, through prompts longer than the
// target's step, a batch carrying a longer prompt before a shorter one, an image piece, and rounds of drafts,
// verifications and partial acceptance
static bool test_device_h(llama_model * model) {
    server_rig host(model, 0.0f, N_MAX, 0, 32, GGML_TYPE_F16, 64);
    auto dev_rig = device_rig(model);
    server_rig & dev = *dev_rig;
    std::mt19937 rng_h(5), rng_d(5);

    const auto same = [&](const char * what, int n_rows) {
        const int want = n_rows >= 64 ? n_rows : 0;
        const int got  = llama_get_nextn_store_rows(dev.tgt.get());
        if (got != want || llama_get_nextn_store_rows(host.tgt.get()) != 0) {
            fprintf(stderr, "%s: %d rows, the store holds %d (want %d), the host rig's %d (want 0)\n", what, n_rows, got,
                    want, llama_get_nextn_store_rows(host.tgt.get()));
            return fail("the store does not hold exactly the rows of a target batch of 64 rows or more");
        }
        if (host.logits != dev.logits) {
            fprintf(stderr, "%s: %d rows\n", what, n_rows);
            return fail("the target's logits moved with the hidden rows on the device");
        }
        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            // a sequence with no row yet keeps none in either rig
            std::vector<uint8_t> st_h, st_d;
            const bool kept_h = common_speculative_get_state(host.spec.get(), s, st_h);
            const bool kept_d = common_speculative_get_state(dev.spec.get(), s, st_d);
            if (kept_h != kept_d || st_h != st_d) {
                fprintf(stderr, "%s: %d rows, seq %d, kept %d and %d\n", what, n_rows, s, kept_h, kept_d);
                return fail("a sequence kept another row with the hidden rows on the device");
            }
            // the head read the same rows, so each sequence's cells in it are the same bytes
            const auto cells = [s](server_rig & rig) {
                llama_context * head = rig.init->context();
                std::vector<uint8_t> out(llama_state_seq_get_size(head, s));
                out.resize(llama_state_seq_get_data(head, out.data(), out.size(), s));
                return out;
            };
            const auto c_h = cells(host), c_d = cells(dev);
            if (c_h != c_d) {
                size_t i = 0;
                while (i < std::min(c_h.size(), c_d.size()) && c_h[i] == c_d[i]) ++i;
                fprintf(stderr, "%s: %d rows, seq %d, the head's cells %zu and %zu bytes, first difference at byte %zu\n",
                        what, n_rows, s, c_h.size(), c_d.size(), i);
                return fail("the head's cells differ with the hidden rows on the device");
            }
        }
        return true;
    };

    const int n_prompt[N_SEQ] = { 150, 70, 200 };
    for (llama_seq_id s = 0; s < N_SEQ; ++s) {
        host.request(s, rng_h, n_prompt[s]);
        dev.request(s, rng_d, n_prompt[s]);
        if (!same("a prompt", n_prompt[s])) return false;
    }
    // two sequences' prompt rows in one batch, the longer first: the equal split interleaves them
    const auto more = host.tokens(rng_h, 90), other = host.tokens(rng_h, 40);
    dev.tokens(rng_d, 130);
    for (server_rig * rig : { &host, &dev }) {
        rig->decode({{ 0, more }, { 1, other }});
        rig->seqs[0].prompt.insert(rig->seqs[0].prompt.end(), more.begin(), more.end());
        rig->seqs[1].prompt.insert(rig->seqs[1].prompt.end(), other.begin(), other.end());
    }
    if (!same("two prompts, the longer first", 130)) return false;
    // an image piece, then the text after it; its rows come from a generator of their own, so both rigs' streams stay
    // in step
    std::vector<float> embd((size_t) 100*llama_model_n_embd_inp(model));
    std::mt19937 rng_img(11);
    std::normal_distribution<float> embd_dis(0.0f, 1.0e-2f);
    for (auto & v : embd) v = embd_dis(rng_img);
    host.image(2, embd, 100);
    dev.image(2, embd, 100);
    if (!same("an image piece", 100)) return false;
    const auto after = host.tokens(rng_h, 6);
    dev.tokens(rng_d, 6);
    host.decode({{ 2, after }});
    dev.decode({{ 2, after }});
    if (!same("the text after an image", 6)) return false;
    for (server_rig * rig : { &host, &dev }) {
        rig->seqs[2].prompt.insert(rig->seqs[2].prompt.end(), after.begin(), after.end());
        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            common_speculative_begin(rig->spec.get(), s, rig->seqs[s].prompt);
        }
    }

    const acceptance acc[N_SEQ] = { {1.0f, 1.0f}, {1.0f, 0.0f}, {0.5f, 0.5f} };
    for (int round = 0; round < 2*WINDOW; ++round) {
        host.draft({0, 1, 2});
        dev.draft({0, 1, 2});
        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            if (host.seqs[s].draft != dev.seqs[s].draft) {
                fprintf(stderr, "round %d, seq %d: drafts", round, s);
                for (const auto t : host.seqs[s].draft) fprintf(stderr, " %d", t);
                fprintf(stderr, " /");
                for (const auto t : dev.seqs[s].draft) fprintf(stderr, " %d", t);
                fprintf(stderr, "\n");
                return fail("the head drafted other tokens with the hidden rows on the device");
            }
        }
        host.verify({2, 0, 1});
        dev.verify({2, 0, 1});
        int n_rows = 0;
        for (llama_seq_id s = 0; s < N_SEQ; ++s) n_rows += 1 + (int) host.seqs[s].draft.size();
        if (!same("a verification", n_rows)) return false;
        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            const int n = acc[s]((int) host.seqs[s].draft.size(), rng_h);
            acc[s]((int) dev.seqs[s].draft.size(), rng_d);
            host.accept(s, n, rng_h);
            dev.accept(s, n, rng_d);
        }
    }
    return true;
}

// with LLAMA_MTP_DEVICE_H=1 only the rows the drafter keeps reach the host: after three sequences' prompts in one
// 200-row batch, at most n_max + 3 rows a sequence (its first n_max + 2 and its last), where the host path copies all
// 200; a 3-row verification, which keeps the host path, reads its 3 rows in both
static bool test_device_h_rows(llama_model * model) {
    server_rig host(model, 0.0f, N_MAX, 0, 32, GGML_TYPE_F16, 64);
    auto dev_rig = device_rig(model);
    server_rig & dev = *dev_rig;
    std::mt19937 rng(6);
    const std::vector<std::pair<llama_seq_id, llama_tokens>> prompts = {
        { 0, host.tokens(rng, 80) }, { 1, host.tokens(rng, 70) }, { 2, host.tokens(rng, 50) } };
    const auto read = [&](server_rig & rig, const std::function<void()> & step) {
        const int64_t before = llama_get_nextn_rows_to_host(rig.tgt.get());
        step();
        return llama_get_nextn_rows_to_host(rig.tgt.get()) - before;
    };
    const int64_t n_host = read(host, [&] { host.decode(prompts); });
    const int64_t n_dev  = read(dev,  [&] { dev.decode(prompts); });
    if (n_host != 200 || n_dev > N_SEQ*(N_MAX + 3)) {
        fprintf(stderr, "a 200-row batch: the host rig read %lld rows (want 200), the device rig %lld (want at most %d)\n",
                (long long) n_host, (long long) n_dev, N_SEQ*(N_MAX + 3));
        return fail("the device rig's target read more than the kept rows back to the host");
    }
    for (server_rig * rig : { &host, &dev }) {
        for (const auto & p : prompts) {
            rig->seqs[p.first].prompt = p.second;
            common_speculative_begin(rig->spec.get(), p.first, p.second);
            rig->seqs[p.first].id_last = 1;
        }
        rig->draft({0});
    }
    const int64_t v_host = read(host, [&] { host.verify({0}); });
    const int64_t v_dev  = read(dev,  [&] { dev.verify({0}); });
    if (v_host != 1 + N_MAX || v_dev != 1 + N_MAX) {
        fprintf(stderr, "a %d-row verification: the host rig read %lld rows, the device rig %lld\n", 1 + N_MAX,
                (long long) v_host, (long long) v_dev);
        return fail("a verification did not read its rows back to the host");
    }
    return true;
}

static bool test_dp(const std::vector<ggml_backend_dev_t> & devices, bool q8_cache) {
    auto baseline = glm_model(1234, {devices.front()});
    auto split = glm_model(1234, devices, LLAMA_SPLIT_MODE_DATA);
    set_env("LLAMA_MTP_DEVICE_H", "1");
    server_rig ref(baseline.get(), 0, N_MAX, 0, 32, GGML_TYPE_F16, 64, true, q8_cache);
    server_rig dp(split.get(), 0, N_MAX, 0, 32, GGML_TYPE_F16, 64, true, q8_cache);
    set_env("LLAMA_MTP_DEVICE_H", nullptr);
    std::mt19937 rng_ref(5), rng_dp(5);
    for (llama_seq_id seq = 0; seq < N_SEQ; ++seq) {
        ref.request(seq, rng_ref, 150 + 17*seq);
        dp.request(seq, rng_dp, 150 + 17*seq);
    }
    size_t greedy_accepted = 0;
    size_t forced_counts[N_MAX + 1] = {};
    for (int round = 0; round < 16; ++round) {
        const std::vector<llama_seq_id> order = round % 2 ? std::vector<llama_seq_id>{1, 2, 0} : std::vector<llama_seq_id>{2, 0, 1};
        ref.draft(order);
        dp.draft(order);
        std::vector<llama_pos> before;
        for (llama_seq_id seq = 0; seq < N_SEQ; ++seq) {
            before.push_back(ref.seqs[seq].n_past);
            if (ref.seqs[seq].draft.size() != N_MAX || dp.seqs[seq].draft.size() != N_MAX) {
                return fail("DP draft round did not produce n_max tokens");
            }
            if (ref.seqs[seq].draft != dp.seqs[seq].draft) { return fail("DP draft tokens differ from the unsplit model"); }
        }
        const auto first = ref.verify(order);
        dp.verify(order);
        const auto top = [](const server_rig & rig, int row) {
            const float * start = rig.logits.data() + row*rig.n_vocab;
            return llama_token(std::max_element(start, start + rig.n_vocab) - start);
        };
        for (llama_seq_id seq : order) {
            int accepted = 0;
            const auto & draft = ref.seqs[seq].draft;
            for (int i = 0; i <= int(draft.size()); ++i) {
                const auto a = top(ref, first.at(seq) + i), b = top(dp, first.at(seq) + i);
                if (a != b) { return fail("DP verification token differs from the unsplit model"); }
                if (i == int(draft.size()) || draft[i] != a) { break; }
                ++accepted;
            }
            greedy_accepted += accepted;
            if (round >= 8) {
                accepted = (round + seq) % (N_MAX + 1);
                ++forced_counts[accepted];
                if (top(ref, first.at(seq) + accepted) != top(dp, first.at(seq) + accepted)) {
                    return fail("DP correction after controlled acceptance differs");
                }
            }
            for (auto * rig : {&ref, &dp}) {
                common_speculative_accept(rig->spec.get(), seq, accepted);
                rig->seqs[seq].n_past = before[seq] + 1 + accepted;
                rig->seqs[seq].id_last = top(*rig, first.at(seq) + accepted);
                if (!llama_memory_seq_rm(llama_get_memory(rig->tgt.get()), seq, rig->seqs[seq].n_past, -1)) {
                    return fail("DP verification rollback failed");
                }
                if (!llama_memory_seq_rm(llama_get_memory(rig->init->context()), seq, rig->seqs[seq].n_past, -1)) {
                    return fail("DP draft cache rollback failed");
                }
            }
        }
        if (round == 7) {
            ref.request(1, rng_ref, 83);
            dp.request(1, rng_dp, 83);
        }
    }
    for (auto count : forced_counts) { if (count == 0) { return fail("DP acceptance coverage is incomplete"); } }
    printf("DP MTP %zu ranks %s: 48 identical draft/verify rounds, %zu greedy matches, controlled 0/1/2 accepts %zu/%zu/%zu\n",
        devices.size(), q8_cache ? "Q8" : "default", greedy_accepted, forced_counts[0], forced_counts[1], forced_counts[2]);
    return true;
}

int main(int argc, char ** argv) {
    common_init();
    common_log_set_verbosity_thold(LOG_LEVEL_WARN);
    llama_log_set(log_errors, nullptr);
    llama_backend_init();

    bool ok = true;
    try {
        if (argc == 3 && strcmp(argv[1], "--dp-device") == 0) {
            ggml_backend_load_all();
            std::vector<ggml_backend_dev_t> devices;
            for (const auto & name : string_split(argv[2], ",")) {
                auto device = ggml_backend_dev_by_name(name.c_str());
                if (!device) { throw std::runtime_error("unknown DP device " + name); }
                devices.push_back(device);
            }
            if (devices.empty()) { throw std::runtime_error("DP devices are empty"); }
            for (size_t count : {2u, 4u}) {
                std::vector<ggml_backend_dev_t> ranks;
                for (size_t i = 0; i < count; ++i) { ranks.push_back(devices[i % devices.size()]); }
                for (const bool q8 : {false, true}) { ok = test_dp(ranks, q8) && ok; }
            }
            llama_backend_free();
            return ok ? 0 : 1;
        }
        if (argc != 1) { throw std::runtime_error("usage: test-speculative-mtp [--dp-device CPU|CUDA0,CUDA1,...]"); }
        auto model = glm_model(1234);
        const std::pair<const char *, std::function<bool(llama_model *)>> tests[] = {
            { "baseline",  test_baseline  },
            { "adaptive",  test_adaptive  },
            { "replay",    test_replay    },
            { "histories", test_histories },
            { "n-min",     test_n_min     },
            { "kept-row",  test_kept_row  },
            { "row-order", test_rows_in_batch_order },
            { "media",     test_media     },
            { "ubatch",    test_draft_ubatch },
            { "indexer",   test_draft_indexer_type },
            { "device-h",  test_device_h  },
            { "device-h-rows", test_device_h_rows },
        };
        for (const auto & t : tests) {
            const bool pass = t.second(model.get());
            printf("%-10s %s\n", t.first, pass ? "OK" : "FAIL");
            ok = ok && pass;
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "error: %s\n", err.what());
        ok = false;
    }
    llama_backend_free();
    return ok ? 0 : 1;
}
