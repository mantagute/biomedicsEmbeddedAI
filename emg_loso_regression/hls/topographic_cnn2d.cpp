#include "topographic_cnn2d.hpp"
#include "cnn2d_weights.h"

namespace {

inline hls_weight_t relu_cast(hls_acc_t val) {
    return (val > hls_acc_t(0)) ? hls_weight_t(val) : hls_weight_t(0);
}

void preprocess_and_map_topography(
    const hls_weight_t x_flat[N_FLAT_FEATURES],
    hls_weight_t grid_in[GRID_ROWS][GRID_COLS][IN_CHANNELS]
) {
    // Mapeia [RMS | MAV | WL] x [EDC(64), FDS(64)] -> (8, 8, 6) com Z-score
    for (int feat = 0; feat < 3; ++feat) {
        for (int g = 0; g < 2; ++g) {
            const int ch = feat * 2 + g;
            const int base = feat * 128 + g * 64;
            for (int r = 0; r < GRID_ROWS; ++r) {
                for (int c = 0; c < GRID_COLS; ++c) {
#pragma HLS PIPELINE II=1
                    const int idx = base + r * GRID_COLS + c;
                    hls_acc_t norm_val = hls_acc_t(x_flat[idx] - STD_MEAN[idx]) * hls_acc_t(STD_INV_SCALE[idx]);
                    grid_in[r][c][ch] = hls_weight_t(norm_val);
                }
            }
        }
    }
}

void conv1_relu(
    const hls_weight_t in_grid[GRID_ROWS][GRID_COLS][IN_CHANNELS],
    hls_weight_t out_grid[GRID_ROWS][GRID_COLS][CONV1_FILTERS]
) {
    for (int r = 0; r < GRID_ROWS; ++r) {
        for (int c = 0; c < GRID_COLS; ++c) {
            for (int f = 0; f < CONV1_FILTERS; ++f) {
#pragma HLS PIPELINE II=1
                hls_acc_t acc = hls_acc_t(CONV1_B[f]);
                for (int ky = 0; ky < 3; ++ky) {
                    const int ir = r + ky - 1;
                    for (int kx = 0; kx < 3; ++kx) {
                        const int ic = c + kx - 1;
                        if (ir >= 0 && ir < GRID_ROWS && ic >= 0 && ic < GRID_COLS) {
                            for (int ch = 0; ch < IN_CHANNELS; ++ch) {
                                acc += hls_acc_t(in_grid[ir][ic][ch]) * hls_acc_t(CONV1_W[ky][kx][ch][f]);
                            }
                        }
                    }
                }
                out_grid[r][c][f] = relu_cast(acc);
            }
        }
    }
}

void maxpool_2x2(
    const hls_weight_t in_grid[GRID_ROWS][GRID_COLS][CONV1_FILTERS],
    hls_weight_t out_grid[POOL1_ROWS][POOL1_COLS][CONV1_FILTERS]
) {
    for (int r = 0; r < POOL1_ROWS; ++r) {
        for (int c = 0; c < POOL1_COLS; ++c) {
            for (int f = 0; f < CONV1_FILTERS; ++f) {
#pragma HLS PIPELINE II=1
                const int r0 = r * 2;
                const int c0 = c * 2;
                hls_weight_t m0 = (in_grid[r0][c0][f] > in_grid[r0][c0 + 1][f]) ? in_grid[r0][c0][f] : in_grid[r0][c0 + 1][f];
                hls_weight_t m1 = (in_grid[r0 + 1][c0][f] > in_grid[r0 + 1][c0 + 1][f]) ? in_grid[r0 + 1][c0][f] : in_grid[r0 + 1][c0 + 1][f];
                out_grid[r][c][f] = (m0 > m1) ? m0 : m1;
            }
        }
    }
}

void conv2_relu_gap(
    const hls_weight_t in_grid[POOL1_ROWS][POOL1_COLS][CONV1_FILTERS],
    hls_weight_t gap_out[CONV2_FILTERS]
) {
    // Funde Conv2D(32, 3x3, same) + ReLU + GlobalAveragePooling2D (4x4 -> >> 4)
    for (int f = 0; f < CONV2_FILTERS; ++f) {
        hls_acc_t sum_gap = 0;
        for (int r = 0; r < POOL1_ROWS; ++r) {
            for (int c = 0; c < POOL1_COLS; ++c) {
#pragma HLS PIPELINE II=1
                hls_acc_t acc = hls_acc_t(CONV2_B[f]);
                for (int ky = 0; ky < 3; ++ky) {
                    const int ir = r + ky - 1;
                    for (int kx = 0; kx < 3; ++kx) {
                        const int ic = c + kx - 1;
                        if (ir >= 0 && ir < POOL1_ROWS && ic >= 0 && ic < POOL1_COLS) {
                            for (int ch = 0; ch < CONV1_FILTERS; ++ch) {
                                acc += hls_acc_t(in_grid[ir][ic][ch]) * hls_acc_t(CONV2_W[ky][kx][ch][f]);
                            }
                        }
                    }
                }
                sum_gap += hls_acc_t(relu_cast(acc));
            }
        }
        // Divisão exata por 16 (4x4) em ponto fixo
        gap_out[f] = hls_weight_t(sum_gap >> 4);
    }
}

void dense_latent_and_head(
    const hls_weight_t gap_out[CONV2_FILTERS],
    hls_weight_t y_angles[N_DOFS]
) {
    hls_weight_t latent[LATENT_DIM];
#pragma HLS ARRAY_PARTITION variable=latent complete dim=1

    for (int j = 0; j < LATENT_DIM; ++j) {
#pragma HLS PIPELINE II=1
        hls_acc_t acc = hls_acc_t(FC_LATENT_B[j]);
        for (int i = 0; i < CONV2_FILTERS; ++i) {
            acc += hls_acc_t(gap_out[i]) * hls_acc_t(FC_LATENT_W[i][j]);
        }
        latent[j] = relu_cast(acc);
    }

    for (int d = 0; d < N_DOFS; ++d) {
#pragma HLS PIPELINE II=1
        hls_acc_t acc = hls_acc_t(REGRESSION_HEAD_B[d]);
        for (int j = 0; j < LATENT_DIM; ++j) {
            acc += hls_acc_t(latent[j]) * hls_acc_t(REGRESSION_HEAD_W[j][d]);
        }
        y_angles[d] = hls_weight_t(acc);
    }
}

} // namespace

void topographic_cnn2d_infer(
    const hls_weight_t x_flat[N_FLAT_FEATURES],
    hls_weight_t y_angles[N_DOFS]
) {
#pragma HLS INTERFACE mode=s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=x_flat bundle=CTRL
#pragma HLS INTERFACE mode=s_axilite port=y_angles bundle=CTRL

    hls_weight_t grid_in[GRID_ROWS][GRID_COLS][IN_CHANNELS];
    hls_weight_t c1_out[GRID_ROWS][GRID_COLS][CONV1_FILTERS];
    hls_weight_t p1_out[POOL1_ROWS][POOL1_COLS][CONV1_FILTERS];
    hls_weight_t gap_out[CONV2_FILTERS];

    preprocess_and_map_topography(x_flat, grid_in);
    conv1_relu(grid_in, c1_out);
    maxpool_2x2(c1_out, p1_out);
    conv2_relu_gap(p1_out, gap_out);
    dense_latent_and_head(gap_out, y_angles);
}
