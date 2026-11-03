// llama-topk-overlap: reads a prompt as the server reads it, step by step, and says for every DSA layer of every step how
// much adjacent rows' indexer selections overlap: the KV a tile of 2, 4, 8 or 16 rows gathers as one union, as a
// multiple of what one row gathers alone (topk-overlap.h). Query-tiled sparse attention is worth building only where
// that multiple stays well under the tile's size, and it is a property of the trained weights, so it is read here on
// the real model.
//
//   llama-topk-overlap -m MODEL -f TEXT [the server's flags: -sm, -ngl, -fa, -b, -ub, -c, cache types]
//
// TOPK_TOKENS caps the prompt at that many tokens (default: all of TEXT). One line a layer a step goes to stdout, then a
// summary over the steps where the cache held more live pools than a row selects.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include "topk-overlap.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static const int64_t TILES[] = {2, 4, 8, 16};

struct layer_total {
    topk_tile_stats st[4];
    int64_t         steps = 0;
};

struct overlap_state {
    int                        step    = -1;
    int                        last_il = 1 << 30;
    int64_t                    pos0    = 0;
    std::map<int, layer_total> layers;
    std::vector<int32_t>       buf;
    bool                       warned  = false;
};

// the live pools the top-k chose among: its input is [the pools' scores, the dump pools], concatenated
static int64_t live_pools(const ggml_tensor * t) {
    const ggml_tensor * k = t->op == GGML_OP_TOP_K ? t : t->src[0];
    if (!k || k->op != GGML_OP_TOP_K) return -1;
    const ggml_tensor * c = k->src[0];
    if (!c || c->op != GGML_OP_CONCAT || !c->src[0]) return -1;
    return c->src[0]->ne[0];
}

static bool on_node(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (overlap_state *) user_data;
    const int il = topk_selection_layer(t->name);
    if (il < 0) return false;
    if (ask) return true;

    if (il <= st->last_il) st->step++;
    st->last_il = il;

    if (t->type != GGML_TYPE_I32) {
        LOG_ERR("%s is %s, not i32: not read\n", t->name, ggml_type_name(t->type));
        return true;
    }
    const int64_t select_k = t->ne[0];
    const int64_t n_rows   = t->ne[1]*t->ne[2]*t->ne[3];
    int64_t       n_live   = live_pools(t);
    if (n_live < 0) {
        if (!st->warned) LOG_WRN("%s: the top-k's input is not [scores, dump pools]; every id counted as live\n", t->name);
        st->warned = true;
        n_live = INT32_MAX;
    }
    st->buf.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, st->buf.data(), 0, ggml_nbytes(t));

    std::printf("step %2d pos %6lld rows %5lld layer %2d pools %6lld select %4lld |", st->step, (long long) st->pos0,
            (long long) n_rows, il, (long long) n_live, (long long) select_k);
    layer_total & lt = st->layers[il];
    const bool sparse = n_live > select_k;
    for (int i = 0; i < 4; ++i) {
        const topk_tile_stats s = topk_tile_union(st->buf.data(), select_k, n_rows, n_live, TILES[i]);
        std::printf(" t%lld %6.3f", (long long) TILES[i], s.mean());
        if (sparse) {
            lt.st[i].tiles        += s.tiles;
            lt.st[i].sum_ratio    += s.sum_ratio;
            lt.st[i].rows_skipped += s.rows_skipped;
        }
    }
    if (sparse) lt.steps++;
    std::printf("%s\n", sparse ? "" : "  (dense: every live pool fits the top-k)");
    std::fflush(stdout);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

    llama_backend_init();
    llama_numa_init(params.numa);

    overlap_state st;
    params.cb_eval           = on_node;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto   llama_init = common_init_from_params(params);
    auto * model      = llama_init->model();
    auto * ctx        = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab *      vocab  = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (const char * cap = std::getenv("TOPK_TOKENS")) {
        const size_t n = (size_t) std::atoll(cap);
        if (n > 0 && n < tokens.size()) tokens.resize(n);
    }
    if (tokens.empty()) {
        LOG_ERR("%s: no prompt tokens (-f TEXT)\n", __func__);
        return 1;
    }
    const size_t n_batch = llama_n_batch(ctx);
    std::printf("tokens %zu, batch %zu, ubatch %u\n", tokens.size(), n_batch, llama_n_ubatch(ctx));

    // one graph a decode: llama_decode splits a batch into -ub graphs, and pos0 labels a whole decode
    const size_t n_step = llama_n_ubatch(ctx);
    for (size_t i = 0; i < tokens.size(); i += n_step) {
        const size_t n = std::min(n_step, tokens.size() - i);
        st.pos0 = (int64_t) i;
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, (int32_t) n))) {
            LOG_ERR("%s: decode failed at token %zu\n", __func__, i);
            return 1;
        }
    }

    std::printf("\nsummary over the steps where the cache held more live pools than a row selects: the KV a tile gathers, "
                "as a multiple of one row's\n");
    topk_tile_stats all[4];
    for (const auto & [il, lt] : st.layers) {
        if (lt.steps == 0) continue;
        std::printf("layer %2d, %lld steps |", il, (long long) lt.steps);
        for (int i = 0; i < 4; ++i) {
            std::printf(" t%lld %6.3f", (long long) TILES[i], lt.st[i].mean());
            all[i].tiles     += lt.st[i].tiles;
            all[i].sum_ratio += lt.st[i].sum_ratio;
        }
        std::printf("\n");
    }
    std::printf("all layers |");
    for (int i = 0; i < 4; ++i) std::printf(" t%lld %6.3f", (long long) TILES[i], all[i].mean());
    std::printf("\n");

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
