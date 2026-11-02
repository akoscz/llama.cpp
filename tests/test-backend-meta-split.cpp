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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr int64_t n_cols = 64;
static constexpr int64_t n_rows = 8;

static int captured = 0;
static ggml_backend_dev_t peer = nullptr;

static ggml_backend_dev_t second_device(ggml_backend_dev_t first) { return peer ? peer : first; }

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
    ggml_backend_dev_t devs[4] = {gpu, second_device(gpu), gpu, second_device(gpu)};
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

static bool test_scatter_views(ggml_backend_dev_t device) {
    ggml_backend_dev_t devices[] = {device, second_device(device), device, second_device(device)};
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devices, 4, split_rows, nullptr), nullptr);
    bool ok = true;
    for (int kind = 0; kind < 3; ++kind) {
        ggml_context * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 8);
        ggml_set_input(input);
        const int64_t rows[] = {1, 0, 2, 5};
        ggml_tensor * owned = ggml_backend_meta_scatter(ctx, input, 1, rows, 4);
        ggml_tensor * view;
        if (kind == 0) {
            view = ggml_reshape_3d(ctx, owned, 16, 2, 8);
            view = ggml_cont(ctx, ggml_permute(ctx, view, 1, 0, 2, 3));
        } else {
            view = ggml_view_2d(ctx, owned, 32, 5, owned->nb[1], 2*owned->nb[1]);
            if (kind == 2) { view = ggml_view_2d(ctx, view, 32, 2, view->nb[1], view->nb[1]); }
        }
        ggml_tensor * out = ggml_scale(ctx, view, 2.0f);
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
        std::vector<float> data(ggml_nelements(input)), result(ggml_nelements(out));
        for (size_t i = 0; i < data.size(); ++i) { data[i] = i + 1; }
        ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        bool case_ok = true;
        for (size_t i = 0; i < result.size(); ++i) {
            const size_t index = kind == 0 ? i/32*32 + i%2*16 + i%32/2 : (kind == 1 ? 64 : 96) + i;
            case_ok = case_ok && result[i] == 2.0f*data[index];
        }
        printf("%s scatter view %d\n", case_ok ? "ok  " : "FAIL", kind);
        ok = case_ok && ok;
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return ok;
}

static bool test_owned_cache_rows(ggml_backend_dev_t device, ggml_type type) {
    ggml_backend_dev_t devices[] = {device, second_device(device), device, second_device(device)};
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devices, 4, split_rows, nullptr), nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_context * weights = ggml_init({ggml_tensor_overhead(), nullptr, true});
    ggml_tensor * cache = ggml_new_tensor_2d(weights, type, 64, 16);
    const int64_t capacity[] = {2, 4, 3, 7};
    const int64_t active[] = {1, 0, 2, 1};
    ggml_backend_meta_split_input(cache, 1, capacity, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
    GGML_ASSERT(buffer);
    std::vector<float> original(64*16), updates(64*4);
    for (size_t i = 0; i < original.size(); ++i) { original[i] = 127.0f*(1 + i/32); }
    for (size_t i = 0; i < updates.size(); ++i) { updates[i] = -127.0f*(1 + i/32); }
    std::vector<uint8_t> bytes(ggml_nbytes(cache));
    ggml_quantize_chunk(type, original.data(), bytes.data(), 0, 16, 64, nullptr);
    ggml_backend_tensor_set(cache, bytes.data(), 0, bytes.size());
    // Checkpoint I/O can touch part of one owner or span several owners.
    const size_t first = cache->nb[1] + 3;
    const size_t count = 9*cache->nb[1] - 7;
    std::vector<uint8_t> checkpoint(count);
    ggml_backend_tensor_get(cache, checkpoint.data(), first, count);
    bool checkpoint_ok = std::equal(checkpoint.begin(), checkpoint.end(), bytes.begin() + first);
    ggml_backend_tensor_memset(cache, 0, first, count);
    ggml_backend_tensor_set(cache, checkpoint.data(), first, count);
    ggml_context * ctx = ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * values = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
    ggml_tensor * writes = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 4);
    ggml_tensor * reads = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    for (ggml_tensor * input : {values, writes, reads}) {
        ggml_set_input(input);
        ggml_backend_meta_split_input(input, input == values ? 1 : 0, active, 4);
    }
    ggml_tensor * stored = ggml_backend_meta_set_rows_local(ctx, cache, values, writes);
    ggml_tensor * out = ggml_backend_meta_get_rows_local(ctx, stored, reads);
    ggml_tensor * full = ggml_cast(ctx, stored, GGML_TYPE_F32);
    ggml_set_output(out);
    ggml_set_output(full);
    bool ok = checkpoint_ok && !ggml_backend_dev_supports_op(ggml_backend_get_device(cpu), stored) &&
              !ggml_backend_dev_supports_op(ggml_backend_get_device(cpu), out);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_build_forward_expand(graph, full);
    ggml_backend_t backends[] = {backend, cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    const int64_t local_writes[] = {1, 0, 2, 6};
    const int32_t local_reads[] = {1, 0, 2, 6};
    const int32_t global_rows[] = {1, 6, 8, 15};
    ggml_backend_tensor_set(values, updates.data(), 0, ggml_nbytes(values));
    ggml_backend_tensor_set(writes, local_writes, 0, ggml_nbytes(writes));
    ggml_backend_tensor_set(reads, local_reads, 0, ggml_nbytes(reads));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(64*4), result_full(64*16);
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    ggml_backend_tensor_get(full, result_full.data(), 0, ggml_nbytes(full));
    for (size_t row = 0; row < 4; ++row) {
        std::copy(updates.begin() + row*64, updates.begin() + (row + 1)*64, original.begin() + global_rows[row]*64);
    }
    ok = ok && result == updates && result_full == original;
    printf("%s owned %s cache reads/writes with unequal capacity and active rows\n", ok ? "ok  " : "FAIL", ggml_type_name(type));
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(weights);
    ggml_backend_free(cpu);
    ggml_backend_free(backend);
    return ok;
}

static bool test_experts(ggml_backend_dev_t gpu, ggml_type type = GGML_TYPE_F32) {
    const int captured_before = captured;
    ggml_backend_dev_t devs[] = {gpu, second_device(gpu), gpu, second_device(gpu)};
    ggml_backend_t meta = ggml_backend_dev_init(ggml_backend_meta_device(devs, 4, split_rows, nullptr), nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    std::vector<float> results[2];
    for (int mode = 0; mode < 2; ++mode) {
        const bool partition = mode == 1;
        ggml_backend_t backend = partition ? meta : cpu;
        ggml_context * weights = ggml_init({8*ggml_tensor_overhead(), nullptr, true});
        ggml_tensor * w[3];
        const int64_t counts[] = {2, 2, 2, 2};
        for (auto & tensor : w) {
            tensor = ggml_new_tensor_3d(weights, type, 256, 256, 8);
            if (partition) { ggml_backend_meta_split_input(tensor, 2, counts, 4); }
        }
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
        GGML_ASSERT(buffer);
        std::vector<float> data(ggml_nelements(w[0]));
        std::vector<uint8_t> quantized(ggml_nbytes(w[0]));
        std::vector<float> importance(256, 1.0f);
        for (int j = 0; j < 3; ++j) {
            for (size_t i = 0; i < data.size(); ++i) { data[i] = float(int((i*13 + j*7) % 31) - 15)/128; }
            if (type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(w[j], data.data(), 0, ggml_nbytes(w[j]));
            } else {
                ggml_quantize_chunk(type, data.data(), quantized.data(), 0, 256*8, 256, importance.data());
                ggml_backend_tensor_set(w[j], quantized.data(), 0, quantized.size());
            }
        }
        ggml_context * ctx = ggml_init({64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 5);
        ggml_set_input(x);
        const int64_t rows[] = {1, 0, 3, 1};
        if (partition) { ggml_backend_meta_split_input(x, 1, rows, 4); }
        ggml_tensor * full = partition ? ggml_backend_meta_gather(ctx, x) : x;
        full = ggml_reshape_3d(ctx, full, 256, 1, 5);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 5);
        ggml_set_input(ids);
        ggml_tensor * routing = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 2, 5);
        ggml_set_input(routing);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, w[0], full, ids);
        ggml_tensor * up = ggml_mul_mat_id(ctx, w[1], full, ids);
        ggml_tensor * out = ggml_mul_mat_id(ctx, w[2], ggml_swiglu_split(ctx, gate, up), ids);
        out = ggml_mul(ctx, out, routing);
        out = ggml_add(ctx, ggml_view_2d(ctx, out, 256, 5, out->nb[2], 0),
                            ggml_view_2d(ctx, out, 256, 5, out->nb[2], out->nb[1]));
        if (partition) { out = ggml_backend_meta_scatter(ctx, ggml_backend_meta_reduce(ctx, out), 1, rows, 4); }
        out = ggml_scale(ctx, out, 0.5f);
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_t backends[] = {backend, cpu};
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, partition ? 2 : 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
        ggml_backend_sched_set_tensor_backend(sched, x, backend);
        ggml_backend_sched_set_tensor_backend(sched, ids, backend);
        ggml_backend_sched_set_tensor_backend(sched, routing, backend);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        std::vector<float> input(256*5);
        for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i % 17) - 8)/16; }
        // The last rank has no selected experts; rows also mix local and remote expert lanes.
        const int32_t selected[] = {0, 1, 2, 3, 1, 4, 3, 0, 5, 2};
        results[mode].resize(6*256*5);
        for (int iteration = 0; iteration < 6; ++iteration) {
            int32_t routed[10];
            float probabilities[10];
            for (int i = 0; i < 10; ++i) { routed[i] = (selected[i] + 2*iteration) % 8; }
            for (int i = 0; i < 10; ++i) { probabilities[i] = float(1 + (i + iteration) % 3)/4; }
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(ids, routed, 0, ggml_nbytes(ids));
            ggml_backend_tensor_set(routing, probabilities, 0, ggml_nbytes(routing));
            GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(out, results[mode].data() + iteration*256*5, 0, ggml_nbytes(out));
        }
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        ggml_free(weights);
    }
    double error = 0, norm = 0;
    for (size_t i = 0; i < results[0].size(); ++i) {
        const double delta = results[1][i] - results[0][i];
        error += delta*delta;
        norm += double(results[0][i])*results[0][i];
    }
    bool ok = std::isfinite(error) && error < (type == GGML_TYPE_F32 ? 1e-6 : 5e-4)*norm;
    if (ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(gpu), "ggml_backend_capture_end")) {
        ok = ok && captured - captured_before == 4;
    }
    printf("%s %s expert partition against unsplit CPU, NMSE %.9g, %d captures\n", ok ? "ok  " : "FAIL", ggml_type_name(type), error/norm, captured - captured_before);
    ggml_backend_free(meta);
    ggml_backend_free(cpu);
    return ok;
}

static bool test_owned_ops(ggml_backend_dev_t gpu, int variant) {
    ggml_backend_dev_t devs[] = {gpu, second_device(gpu), gpu, second_device(gpu)};
    ggml_backend_t meta = ggml_backend_dev_init(ggml_backend_meta_device(devs, 4, split_rows, nullptr), nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    std::vector<float> results[2];
    const int64_t owners[] = {1, 0, 2, 1};
    for (int mode = 0; mode < 2; ++mode) {
        ggml_backend_t backend = mode ? meta : cpu;
        ggml_context * ctx = ggml_init({32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        std::vector<ggml_tensor *> inputs;
        std::vector<ggml_tensor *> outputs;
        auto input = [&](ggml_type type, std::array<int64_t, 4> ne, int axis) {
            ggml_tensor * t = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_input(t);
            if (mode && axis >= 0) { ggml_backend_meta_split_input(t, axis, owners, 4); }
            inputs.push_back(t);
            return t;
        };
        ggml_tensor * out;
        if (variant == 7) {
            auto * q = input(GGML_TYPE_F32, {128, 2, 4, 1}, 2);
            auto * positions = input(GGML_TYPE_I32, {4, 1, 1, 1}, 0);
            out = ggml_rope(ctx, q, positions, 64, 0);
        } else if (variant == 8) {
            auto * x = input(GGML_TYPE_F32, {6, 128, 4, 1}, 2);
            auto * w = input(GGML_TYPE_F32, {4, 128, 1, 1}, -1);
            out = ggml_ssm_conv(ctx, x, w);
        } else if (variant == 9) {
            auto * x = input(GGML_TYPE_F32, {128, 3, 2, 4}, 3);
            auto * w = input(GGML_TYPE_F32, {128, 64, 2, 1}, -1);
            out = ggml_mul_mat(ctx, w, x);
        } else if (variant == 10) {
            auto * cache = input(GGML_TYPE_F32, {128, 64, 4, 1}, 2);
            auto * indices = input(GGML_TYPE_I32, {8, 4, 1, 1}, 1);
            out = ggml_get_rows(ctx, cache, indices);
        } else if (variant >= 5) {
            const int snapshots = variant == 5 ? 1 : 3;
            ggml_tensor * q = input(GGML_TYPE_F32, {128, 2, 3, 4}, 3);
            ggml_tensor * k = input(GGML_TYPE_F32, {128, 2, 3, 4}, 3);
            ggml_tensor * v = input(GGML_TYPE_F32, {128, 2, 3, 4}, 3);
            ggml_tensor * g = input(GGML_TYPE_F32, {128, 2, 3, 4}, 3);
            ggml_tensor * beta = input(GGML_TYPE_F32, {1, 2, 3, 4}, 3);
            ggml_tensor * state = input(GGML_TYPE_F32, {128, 128, 2, 4}, 3);
            out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, snapshots);
            outputs.push_back(ggml_scale(ctx, ggml_view_4d(ctx, out, 128, 2, 3, 4,
                128*sizeof(float), 256*sizeof(float), 3*256*sizeof(float), 0), 2.0f));
            outputs.push_back(ggml_scale(ctx, ggml_view_3d(ctx, out, 128*128*2, 4, snapshots,
                128*128*2*sizeof(float), 4*128*128*2*sizeof(float), 3*4*256*sizeof(float)), 2.0f));
        } else if (variant == 3) {
            ggml_tensor * mixes = input(GGML_TYPE_F32, {24, 4, 1, 1}, 1);
            ggml_tensor * scale = input(GGML_TYPE_F32, {3, 1, 1, 1}, -1);
            ggml_tensor * base = input(GGML_TYPE_F32, {24, 1, 1, 1}, -1);
            ggml_tensor * x = input(GGML_TYPE_F32, {64, 4, 4, 1}, 2);
            ggml_tensor * weights = ggml_dsv4_hc_weights(ctx, mixes, scale, base, 1e-6f, 4);
            ggml_tensor * pre = ggml_view_2d(ctx, weights, 4, 4, weights->nb[1], 0);
            ggml_tensor * post = ggml_view_2d(ctx, weights, 4, 4, weights->nb[1], 4*sizeof(float));
            ggml_tensor * comb = ggml_dsv4_hc_comb(ctx, mixes, scale, base, 1e-6f, 4);
            out = ggml_dsv4_hc_post(ctx, ggml_dsv4_hc_pre(ctx, x, pre), x, post, comb);
        } else if (variant == 4) {
            ggml_tensor * cells = input(GGML_TYPE_I32, {64, 2, 1, 4}, 3);
            ggml_tensor * rows = input(GGML_TYPE_I32, {4, 3, 4, 1}, 2);
            out = ggml_cast(ctx, ggml_kpool_mask(ctx, cells, rows, GGML_KPOOL_MASK_SEL, 0), GGML_TYPE_F32);
        } else {
            const int64_t width = variant == 1 ? 512 : 128;
            const int64_t cells = variant == 1 ? 256 : 64;
            const int64_t heads = variant == 1 ? 8 : variant == 2 ? 32 : 2;
            ggml_tensor * q = input(GGML_TYPE_F32, variant == 2 ? std::array<int64_t, 4>{width, heads, 3, 4} : std::array<int64_t, 4>{width, 3, heads, 4}, 3);
            ggml_tensor * k = input(GGML_TYPE_F16, variant == 2 ? std::array<int64_t, 4>{width, 1, cells, 4} : std::array<int64_t, 4>{width, cells, 1, 4}, 3);
            if (variant == 2) {
                ggml_tensor * w = input(GGML_TYPE_F32, {heads, 3, 1, 4}, 3);
                ggml_tensor * mask = input(GGML_TYPE_F16, {8, 3, 1, 4}, 3);
                ggml_tensor * rows = input(GGML_TYPE_I32, {8, 4, 1, 1}, 1);
                out = ggml_lightning_indexer_rows(ctx, q, k, rows, w, mask);
            } else {
                ggml_tensor * v = input(GGML_TYPE_F16, {width, cells, 1, 4}, 3);
                if (variant == 1) {
                    ggml_tensor * ids = input(GGML_TYPE_I32, {32, 3, 1, 4}, 3);
                    out = ggml_flash_attn_ext_sparse(ctx, q, k, v, ids, 0.125f);
                } else {
                    ggml_tensor * mask = input(GGML_TYPE_F16, {64, 32, 1, 4}, 3);
                    out = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
                }
            }
        }
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        for (auto t : outputs) { ggml_set_output(t); ggml_build_forward_expand(graph, t); }
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
        for (auto t : inputs) {
            std::vector<float> f32(ggml_nelements(t));
            for (size_t i = 0; i < f32.size(); ++i) { f32[i] = float(int(i % 23) - 11)/32; }
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, f32.data(), 0, ggml_nbytes(t));
            } else if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16(f32.size());
                ggml_fp32_to_fp16_row(f32.data(), f16.data(), f16.size());
                ggml_backend_tensor_set(t, f16.data(), 0, ggml_nbytes(t));
            } else {
                std::vector<int32_t> ids(f32.size());
                for (size_t i = 0; i < ids.size(); ++i) { ids[i] = variant == 1 ? 2*(i % 32) + (i/32) % 2 : 7*(i % 8) + (i/8) % 8; }
                ggml_backend_tensor_set(t, ids.data(), 0, ggml_nbytes(t));
            }
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        results[mode].resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, results[mode].data(), 0, ggml_nbytes(out));
        for (auto t : outputs) {
            const size_t offset = results[mode].size();
            results[mode].resize(offset + ggml_nelements(t));
            ggml_backend_tensor_get(t, results[mode].data() + offset, 0, ggml_nbytes(t));
        }
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }
    double error = 0, norm = 0;
    for (size_t i = 0; i < results[0].size(); ++i) {
        if (results[0][i] == results[1][i] && !std::isfinite(results[0][i])) { continue; }
        const double delta = results[1][i] - results[0][i];
        error += delta*delta;
        norm += double(results[0][i])*results[0][i];
    }
    const bool ok = variant == 4 ? results[0] == results[1] : std::isfinite(error) && error < 1e-6*norm;
    const char * names[] = {"attention", "sparse attention", "indexer", "hyper-connections", "k-pool mask", "KDA final state", "KDA snapshots", "RoPE", "SSM convolution", "batched projection", "batched cache gather"};
    printf("%s owned %s, NMSE %.9g\n", ok ? "ok  " : "FAIL", names[variant], norm ? error/norm : error);
    ggml_backend_free(meta);
    ggml_backend_free(cpu);
    return ok;
}

static bool test_sequence_axis(ggml_backend_dev_t gpu) {
    const int captured_before = captured;
    std::array<int64_t, 4> rows = {1, 0, 2, 5};
    ggml_backend_dev_t devs[4] = {gpu, second_device(gpu), gpu, second_device(gpu)};
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

int main(int argc, char ** argv) {
    // Isolate partition correctness from the optional BF16 collective round trip.
#ifdef _WIN32
    _putenv_s("GGML_CUDA_AR_BF16_THRESHOLD", "0");
#else
    setenv("GGML_CUDA_AR_BF16_THRESHOLD", "0", 1);
#endif
    ggml_log_set(log_cb, nullptr);
    ggml_backend_dev_t gpu = nullptr;
    if (argc == 3 && strcmp(argv[1], "--owned-op-cpu") == 0) {
        return test_owned_ops(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), atoi(argv[2])) ? 0 : 1;
    }
    if (argc > 1 && strcmp(argv[1], "--cpu") == 0) {
        gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        bool ok = true;
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_IQ3_XXS}) { ok = test_experts(gpu, type) && ok; }
        for (int variant = 0; variant < 11; ++variant) { ok = test_owned_ops(gpu, variant) && ok; }
        ok = test_scatter_views(gpu) && ok;
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) { ok = test_owned_cache_rows(gpu, type) && ok; }
        return ok ? 0 : 1;
    }
    for (size_t i = 0; i < ggml_backend_dev_count() && gpu == nullptr; i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = ggml_backend_dev_get(i);
        }
    }
    if (gpu == nullptr) {
        printf("no GPU for the meta backend, skipped\n");
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--two-gpu") == 0) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            if (dev != gpu && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) { peer = dev; break; }
        }
        if (!peer) { printf("need two physical GPUs, skipped\n"); return 77; }
        printf("four logical ranks on two physical GPUs; not four-card qualification\n");
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
    ok = test_scatter_views(gpu) && ok;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) { ok = test_owned_cache_rows(gpu, type) && ok; }
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_IQ3_XXS}) { ok = test_experts(gpu, type) && ok; }
    for (int variant = 0; variant < 11; ++variant) { ok = test_owned_ops(gpu, variant) && ok; }

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
