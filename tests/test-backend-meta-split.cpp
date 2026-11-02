// The meta backend (-sm tensor) gives each device its slice of a split tensor: a tensor with the split axis cut to the
// slice, and the strides of the dims outside the split axis scaled with it. The dims outside were those with a greater
// stride than the split axis. A tensor split along an axis of extent 1, as one head split two ways, has the same stride
// on that axis as on the dim after it, so the dim after it kept its full stride: the device with no slice had a tensor
// of 0 bytes that took a whole chunk of every write, and writing the tensor aborted (tensor write out of bounds).
//
// Each tensor below is written, read back and cleared through the meta buffer, split along axis 2 as its name says:
// heads_<a><b> has a heads on the first device and b on the second. The meta backend is over the first GPU twice, which
// splits over two devices as two GPUs do; with no GPU the test is skipped.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int64_t n_cols = 64;
static constexpr int64_t n_rows = 8;

static int captured = 0;

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (strstr(text, "ggml_backend_cuda_capture_end") && strstr(text, " nodes, ")) { ++captured; }
    if (level != GGML_LOG_LEVEL_DEBUG) { fputs(text, stderr); }
}

static ggml_backend_meta_split_state split_by_name(const ggml_tensor * tensor, void * /*userdata*/) {
    GGML_ASSERT(strncmp(tensor->name, "heads_", 6) == 0);
    ggml_backend_meta_split_state split_state = {GGML_BACKEND_SPLIT_AXIS_2, {0}, {1}, 1};
    split_state.ne[0] = tensor->name[6] - '0';
    split_state.ne[1] = tensor->name[7] - '0';
    GGML_ASSERT(split_state.ne[0] + split_state.ne[1] == tensor->ne[2]);
    return split_state;
}

static ggml_backend_meta_split_state split_rows(const ggml_tensor * tensor, void * userdata) {
    if (strcmp(tensor->name, "tokens") == 0) {
        ggml_backend_meta_split_state state = {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
        const auto & rows = *static_cast<const std::array<int64_t, 4> *>(userdata);
        std::copy(rows.begin(), rows.end(), state.ne);
        return state;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static bool test_rows(ggml_backend_dev_t gpu, bool explicit_layout) {
    std::array<int64_t, 4> rows = {1, 0, 2, 5};
    ggml_backend_dev_t devs[4] = {gpu, gpu, gpu, gpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 4, split_rows, &rows);
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
    ggml_context * weights = ggml_init({4*ggml_tensor_overhead(), nullptr, true});
    ggml_tensor * table = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 32, 16);
    ggml_context * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * tokens = ggml_new_tensor_1d(explicit_layout ? ctx : weights, GGML_TYPE_I32, 8);
    ggml_set_name(tokens, "tokens");
    ggml_set_input(tokens);
    if (explicit_layout) { ggml_backend_meta_split_input(tokens, 0, rows.data(), rows.size()); }
    ggml_tensor * projection = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 32, 32);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights, ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(buffer);

    ggml_tensor * embd = ggml_get_rows(ctx, table, tokens);
    ggml_tensor * out = ggml_mul_mat(ctx, projection, embd);
    ggml_tensor * gathered = nullptr;
    if (explicit_layout) {
        gathered = ggml_backend_meta_gather(ctx, out);
        ggml_set_output(gathered);
        const int64_t next_rows[] = {0, 3, 4, 1};
        out = ggml_backend_meta_scatter(ctx, gathered, 1, next_rows, 4);
        out = ggml_mul_mat(ctx, projection, out);
    }
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
    std::vector<float> table_data(32*16), projection_data(32*32, 0.0f), result(32*8);
    for (int i = 0; i < 32*16; ++i) { table_data[i] = i + 1; }
    for (int i = 0; i < 32; ++i) { projection_data[32*i + i] = 2.0f; }
    const int32_t token_data[8] = {15, 2, 9, 0, 7, 1, 13, 4};
    ggml_backend_tensor_set(table, table_data.data(), 0, ggml_nbytes(table));
    ggml_backend_tensor_set(tokens, token_data, 0, ggml_nbytes(tokens));
    ggml_backend_tensor_set(projection, projection_data.data(), 0, ggml_nbytes(projection));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    bool ok = true;
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 32; ++col) {
            ok = ok && result[32*row + col] == (explicit_layout ? 4.0f : 2.0f)*table_data[32*token_data[row] + col];
        }
    }
    if (gathered) {
        ggml_backend_tensor_get(gathered, result.data(), 0, ggml_nbytes(gathered));
        for (int row = 0; row < 8; ++row) {
            for (int col = 0; col < 32; ++col) {
                ok = ok && result[32*row + col] == 2.0f*table_data[32*token_data[row] + col];
            }
        }
    }
    printf("%s four-rank row split with an empty rank%s\n", ok ? "ok  " : "FAIL", explicit_layout ? ", gathered and redistributed" : "");
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(weights);
    ggml_backend_free(backend);
    return ok;
}

static bool test_sequence_axis(ggml_backend_dev_t gpu) {
    const int captured_before = captured;
    std::array<int64_t, 4> rows = {1, 0, 2, 5};
    ggml_backend_dev_t devs[4] = {gpu, gpu, gpu, gpu};
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devs, 4, split_rows, &rows), nullptr);
    ggml_context * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 2, 2, 8);
    ggml_backend_meta_split_input(x, 3, rows.data(), rows.size());
    ggml_set_input(x);
    ggml_tensor * gathered = ggml_backend_meta_gather(ctx, ggml_scale(ctx, x, 2.0f));
    const int64_t next_rows[] = {0, 0, 0, 8};
    ggml_tensor * out = ggml_scale(ctx, ggml_backend_meta_scatter(ctx, gathered, 3, next_rows, 4), 2.0f);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = {backend, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_set_tensor_backend(sched, x, backend);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    std::vector<float> data(ggml_nelements(x)), result(data.size());
    bool ok = true;
    for (int iteration = 0; iteration < 6; ++iteration) {
        for (size_t i = 0; i < data.size(); ++i) { data[i] = float(i + iteration); }
        ggml_backend_tensor_set_async(backend, x, data.data(), 0, ggml_nbytes(x));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(backend, out, result.data(), 0, ggml_nbytes(out));
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < data.size(); ++i) { ok = ok && result[i] == 4.0f*data[i]; }
    }
    ggml_backend_tensor_memset(x, 0, 0, ggml_nbytes(x));
    ggml_backend_tensor_get(x, result.data(), 0, ggml_nbytes(x));
    ok = ok && result == std::vector<float>(data.size(), 0.0f);
    if (ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(gpu), "ggml_backend_capture_end")) {
        ok = ok && captured - captured_before == 4;
    }
    printf("%s sequence-axis gather/scatter, %d captures and async transfers\n", ok ? "ok  " : "FAIL", captured - captured_before);
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(backend);
    ggml_backend_free(cpu);
    return ok;
}

int main() {
    ggml_log_set(log_cb, nullptr);
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && gpu == nullptr; i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = ggml_backend_dev_get(i);
        }
    }
    if (gpu == nullptr) {
        printf("no GPU for the meta backend, skipped\n");
        return 0;
    }
    ggml_backend_dev_t devs[2] = {gpu, gpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, split_by_name, nullptr);

    const char * names[] = {"heads_10", "heads_01", "heads_11"};
    const ggml_init_params params = {
        /*.mem_size   =*/ 3*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    std::vector<ggml_tensor *> tensors;
    for (const char * name : names) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_cols, n_rows, (name[6] - '0') + (name[7] - '0'));
        ggml_set_name(t, name);
        tensors.push_back(t);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(buf != nullptr);

    bool ok = true;
    for (ggml_tensor * t : tensors) {
        std::vector<float> data(ggml_nelements(t));
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = 1.0f + i;
        }
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        std::vector<float> back(data.size());
        ggml_backend_tensor_get(t, back.data(), 0, ggml_nbytes(t));
        bool ok_t = back == data;

        ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
        ggml_backend_tensor_get(t, back.data(), 0, ggml_nbytes(t));
        ok_t = ok_t && back == std::vector<float>(data.size(), 0.0f);

        printf("%s %s\n", ok_t ? "ok  " : "FAIL", t->name);
        ok = ok && ok_t;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    ok = test_rows(gpu, false) && ok;
    ok = test_rows(gpu, true) && ok;
    ok = test_sequence_axis(gpu) && ok;

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
