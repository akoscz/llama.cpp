// The meta backend (-sm tensor) gives each tensor of its buffers a tensor on every device it splits over. A view made
// of a tensor uses the buffer of that tensor, so a graph's views of the weights have their device tensors in the
// weights' buffer, beside the views of every other graph over the same weights. Those device tensors were kept in two
// containers that took turns: computing a graph that had not been computed last reset the container that did not hold
// its own tensors, though it could hold another graph's. A graph computed again without being allocated again then
// used device tensors that had been freed, and given to the next views made:
//
// 1. Two backends, as a draft context beside the target: graph A (y = the first half of w's rows, viewed, plus x) is
//    computed on one meta backend, then two other graphs over w, each with a view of other rows, on another. A is
//    computed again on its backend, which reuses the device graphs it built for A, and its view of w read the rows of
//    the last view made.
// 2. One backend, as graph slots: graph A is computed, then graph B is allocated by a scheduler of its own, and the two
//    are computed in turn on the same meta backend, which builds its device graphs again for each. Computing B freed
//    the device tensors of A's view, and building A's device graphs again aborted.
//
// Each graph's output must be its own rows of w plus its x, the values changed at every compute. The meta backend is
// over the first GPU twice, which splits over two devices as two GPUs do; with no GPU the test is skipped.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int64_t n_cols = 64;
static constexpr int64_t n_rows = 8;

static ggml_backend_meta_split_state mirrored(const ggml_tensor * /*tensor*/, void * /*userdata*/) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

// y = rows [row0, row0 + n_rows/2) of w, viewed, plus x; allocated and computed by a scheduler of its own
struct view_graph {
    int64_t              row0  = 0;
    ggml_context       * ctx   = nullptr;
    ggml_cgraph        * gf    = nullptr;
    ggml_tensor        * x     = nullptr;
    ggml_tensor        * y     = nullptr;
    ggml_backend_sched_t sched = nullptr;

    view_graph(ggml_backend_t backend, ggml_backend_t cpu, ggml_tensor * w, int64_t row0) : row0(row0) {
        const ggml_init_params params = {
            /*.mem_size   =*/ 8*ggml_tensor_overhead() + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        ggml_tensor * v = ggml_view_2d(ctx, w, n_cols, n_rows/2, w->nb[1], row0*w->nb[1]);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_cols, n_rows/2);
        ggml_set_input(x);
        y = ggml_add(ctx, v, x);
        ggml_set_output(y);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);

        ggml_backend_t backends[2] = {backend, cpu};
        sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, gf));
    }

    ~view_graph() {
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    // computes y for an x of `step`, allocated or not, and checks it
    bool compute(const std::vector<float> & w_data, float step, const char * what) {
        std::vector<float> x_data(n_cols*n_rows/2);
        for (size_t i = 0; i < x_data.size(); i++) {
            x_data[i] = step + 0.25f*i;
        }
        ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
        std::vector<float> y_data(x_data.size());
        ggml_backend_tensor_get(y, y_data.data(), 0, ggml_nbytes(y));
        for (size_t i = 0; i < y_data.size(); i++) {
            const float expected = w_data[row0*n_cols + i] + x_data[i];
            if (y_data[i] != expected) {
                printf("FAIL %s: y[%zu] = %g, expected %g (the rows from %lld of w plus x)\n",
                    what, i, y_data[i], expected, (long long) row0);
                return false;
            }
        }
        printf("ok   %s\n", what);
        return true;
    }
};

static bool test_owned_cache_views(ggml_backend_dev_t gpu, ggml_type type) {
    ggml_backend_dev_t devices[] = {gpu, gpu, gpu, gpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devices, 4, mirrored, nullptr);
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = {backend, cpu};
    ggml_context * weights = ggml_init({ggml_tensor_overhead(), nullptr, true});
    ggml_tensor * cache = ggml_new_tensor_3d(weights, type, 64, 16, 8);
    const int64_t streams[] = {1, 0, 2, 5};
    ggml_backend_meta_split_input(cache, 2, streams, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights, ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(buffer);
    bool ok = ggml_backend_buffer_get_size(buffer) < ggml_nbytes(cache);
    std::vector<float> data(ggml_nelements(cache));
    for (size_t i = 0; i < data.size(); ++i) { data[i] = 127.0f*(1 + i/32); }
    std::vector<uint8_t> bytes(ggml_nbytes(cache));
    GGML_ASSERT(ggml_quantize_chunk(type, data.data(), bytes.data(), 0, 16*8, 64, nullptr) == bytes.size());
    ggml_backend_tensor_set(cache, bytes.data(), 0, bytes.size());
    for (int kind = 0; kind < 3; ++kind) {
        ggml_context * ctx = ggml_init({8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        ggml_tensor * view;
        if (kind == 1) {
            view = ggml_view_4d(ctx, cache, 32, 2, 7, 4, ggml_row_size(type, 32), cache->nb[1], cache->nb[2], cache->nb[2] + 2*cache->nb[1]);
        } else {
            view = ggml_view_2d(ctx, cache, 64, 7, cache->nb[1], 5*cache->nb[2] + 3*cache->nb[1]);
            if (kind == 2) { view = ggml_view_1d(ctx, view, 128, ggml_row_size(type, 64)); }
        }
        ggml_tensor * out = ggml_scale(ctx, ggml_cast(ctx, view, GGML_TYPE_F32), 2.0f);
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        std::vector<float> result(ggml_nelements(out));
        bool case_ok = true;
        for (int step = 0; step < 4; ++step) {
            GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
            for (size_t i = 0; i < result.size(); ++i) {
                const size_t src = kind == 1 ? ((1 + i/(64*7))*16 + 2)*64 + i%(64*7) : (5*16 + (kind == 2 ? 4 : 3))*64 + i;
                case_ok = case_ok && result[i] == 2.0f*data[src];
            }
        }
        printf("%s owned %s cache view %d: %zu values\n", case_ok ? "ok  " : "FAIL", ggml_type_name(type), kind, result.size());
        ok = case_ok && ok;
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    ggml_backend_buffer_free(buffer);
    ggml_free(weights);
    return ok;
}

static bool test_mirrored_copies(ggml_backend_dev_t device) {
    ggml_backend_dev_t devices[] = {device, device};
    auto meta = ggml_backend_meta_device(devices, 2, mirrored, nullptr);
    auto src_backend = ggml_backend_dev_init(meta, nullptr);
    auto dst_backend = ggml_backend_dev_init(meta, nullptr);
    auto src_ctx = ggml_init({4*ggml_tensor_overhead(), nullptr, true});
    auto dst_ctx = ggml_init({ggml_tensor_overhead(), nullptr, true});
    auto * base = ggml_new_tensor_2d(src_ctx, GGML_TYPE_I32, 64, 4);
    auto * view = ggml_view_2d(src_ctx, base, 64, 2, base->nb[1], base->nb[1]);
    auto * dst = ggml_new_tensor_2d(dst_ctx, GGML_TYPE_I32, 64, 2);
    auto src_buf = ggml_backend_alloc_ctx_tensors_from_buft(src_ctx, ggml_backend_dev_buffer_type(meta));
    auto dst_buf = ggml_backend_alloc_ctx_tensors_from_buft(dst_ctx, ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(src_buf && dst_buf);
    std::vector<int32_t> values(256), got(128);
    bool ok = true;
    for (int step = 0; step < 4; ++step) {
        for (size_t i = 0; i < values.size(); ++i) { values[i] = (int32_t) (step*1000 + i); }
        ggml_backend_tensor_set_async(src_backend, base, values.data(), 0, ggml_nbytes(base));
        ggml_backend_tensor_copy_async(src_backend, dst_backend, view, dst);
        ggml_backend_tensor_get_async(dst_backend, dst, got.data(), 0, ggml_nbytes(dst));
        ggml_backend_synchronize(dst_backend);
        for (size_t i = 0; i < got.size(); ++i) { ok &= got[i] == values[64 + i]; }
        // The direct buffer route must exist too; a host-staged fallback is not this contract.
        ok &= ggml_backend_buffer_copy_tensor(view, dst);
    }
    ggml_backend_synchronize(src_backend);
    ggml_backend_buffer_free(src_buf);
    ggml_free(src_ctx);
    ggml_backend_tensor_get(dst, got.data(), 0, ggml_nbytes(dst));
    for (size_t i = 0; i < got.size(); ++i) { ok &= got[i] == values[64 + i]; }
    ggml_backend_buffer_free(dst_buf);
    ggml_free(dst_ctx);
    ggml_backend_free(dst_backend);
    ggml_backend_free(src_backend);
    printf("%s mirrored owned row copies after source release\n", ok ? "ok  " : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    const bool cpu_only = argc == 2 && strcmp(argv[1], "--cpu") == 0;
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && gpu == nullptr; i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) ==
                (cpu_only ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU)) {
            gpu = ggml_backend_dev_get(i);
        }
    }
    if (gpu == nullptr) {
        printf("no GPU for the meta backend, skipped\n");
        return 0;
    }
    ggml_backend_dev_t devs[2] = {gpu, gpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, mirrored, nullptr);

    const ggml_init_params params_w = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_w = ggml_init(params_w);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_cols, n_rows);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(buf_w != nullptr);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> w_data(n_cols*n_rows);
    for (size_t i = 0; i < w_data.size(); i++) {
        w_data[i] = 1000.0f*(i/n_cols) + i%n_cols;
    }
    ggml_backend_tensor_set(w, w_data.data(), 0, ggml_nbytes(w));

    ggml_backend_t cpu       = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backend_a = ggml_backend_dev_init(meta, nullptr);
    ggml_backend_t backend_b = ggml_backend_dev_init(meta, nullptr);

    bool ok = test_mirrored_copies(gpu);
    {
        view_graph a(backend_a, cpu, w, 0);
        ok = a.compute(w_data, 1.0f, "two backends: A") && ok;
        {
            view_graph b(backend_b, cpu, w, n_rows/2);
            ok = b.compute(w_data, 2.0f, "two backends: B on the other backend") && ok;
        }
        {
            view_graph c(backend_b, cpu, w, n_rows/4);
            ok = c.compute(w_data, 3.0f, "two backends: C, allocated anew on the other backend") && ok;
        }
        ok = a.compute(w_data, 4.0f, "two backends: A again, not allocated again") && ok;
    }
    {
        view_graph a(backend_a, cpu, w, 0);
        ok = a.compute(w_data, 5.0f, "one backend: A") && ok;
        view_graph b(backend_a, cpu, w, n_rows/2);
        for (int i = 0; i < 3; i++) {
            ok = b.compute(w_data, 10.0f*i + 6.0f, "one backend: B, allocated after A was computed") && ok;
            ok = a.compute(w_data, 10.0f*i + 7.0f, "one backend: A again, after B") && ok;
        }
    }

    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    ggml_backend_free(cpu);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);

    ok = test_owned_cache_views(gpu, GGML_TYPE_F32) && ok;
    ok = test_owned_cache_views(gpu, GGML_TYPE_Q8_0) && ok;

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
