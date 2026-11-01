#pragma once

// How much adjacent prompt rows' indexer selections overlap. Query-tiled sparse attention gathers, for a tile of
// adjacent rows, the union of the cells each row selected, once, with a mask per row. What that tile loads, as a
// multiple of what one row loads alone, is the tile's union of selected pools over the mean number its rows selected:
// 1 when every row picked the same pools, the tile's size when no two rows share one. The cells of a pool are its own,
// so the multiple is the same counted in pools or in cells.

#include <algorithm>
#include <cstdint>
#include <vector>

struct topk_tile_stats {
    int64_t tiles        = 0;   // tiles counted
    double  sum_ratio    = 0.0; // the sum over them of the tile's load as a multiple of one row's
    int64_t rows_skipped = 0;   // rows in no counted tile: past the last whole tile, or in a tile with a row of no live pool

    double mean() const { return tiles ? sum_ratio / tiles : 0.0; }
};

// sel holds n_rows rows of select_k pool ids, row r at sel[r*select_k]. An id at or past n_live is a dump pool, which
// the top-k takes only when fewer than select_k pools are live, and which holds no cell the tile gathers. Tiles are rows
// [i*tile, (i+1)*tile).
inline topk_tile_stats topk_tile_union(const int32_t * sel, int64_t select_k, int64_t n_rows, int64_t n_live, int64_t tile) {
    topk_tile_stats st;
    if (tile <= 0) return st;

    std::vector<int32_t> uni;
    std::vector<int32_t> row;
    for (int64_t first = 0; first + tile <= n_rows; first += tile) {
        uni.clear();
        int64_t live    = 0;
        bool    counted = true;
        for (int64_t r = first; r < first + tile; ++r) {
            row.clear();
            for (int64_t j = 0; j < select_k; ++j) {
                const int32_t id = sel[r*select_k + j];
                if (id >= 0 && id < n_live) row.push_back(id);
            }
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            if (row.empty()) counted = false;
            live += (int64_t) row.size();
            uni.insert(uni.end(), row.begin(), row.end());
        }
        if (!counted) {
            st.rows_skipped += tile;
            continue;
        }
        std::sort(uni.begin(), uni.end());
        const int64_t n_uni = std::unique(uni.begin(), uni.end()) - uni.begin();
        st.tiles++;
        st.sum_ratio += (double) n_uni * (double) tile / (double) live;
    }
    st.rows_skipped += n_rows % tile;
    return st;
}
