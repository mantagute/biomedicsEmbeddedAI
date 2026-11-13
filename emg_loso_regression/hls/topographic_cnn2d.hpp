#ifndef TOPOGRAPHIC_CNN2D_HPP
#define TOPOGRAPHIC_CNN2D_HPP

#include <ap_fixed.h>

// Dimensões físicas do pipeline HD-sEMG (2 arrays 8x8 x 3 features = 384)
constexpr int N_FLAT_FEATURES = 384;
constexpr int GRID_ROWS = 8;
constexpr int GRID_COLS = 8;
constexpr int IN_CHANNELS = 6;
constexpr int CONV1_FILTERS = 16;
constexpr int POOL1_ROWS = 4;
constexpr int POOL1_COLS = 4;
constexpr int CONV2_FILTERS = 32;
constexpr int LATENT_DIM = 32;
constexpr int N_DOFS = 5;

// Precisão validada em simulação bit-accurate (sem saturação nos ângulos até +-128°)
typedef ap_fixed<16, 8, AP_RND, AP_SAT> hls_weight_t;
typedef ap_fixed<32, 16, AP_RND, AP_SAT> hls_acc_t;

// Top-level synthesizable function para Vitis HLS (Zybo Z7-20 / ZCU104)
void topographic_cnn2d_infer(
    const hls_weight_t x_flat[N_FLAT_FEATURES],
    hls_weight_t y_angles[N_DOFS]
);

#endif // TOPOGRAPHIC_CNN2D_HPP
