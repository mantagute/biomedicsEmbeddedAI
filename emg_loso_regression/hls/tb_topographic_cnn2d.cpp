#include <cmath>
#include <iostream>
#include "topographic_cnn2d.hpp"
#include "golden_sample.h"

int main() {
    hls_weight_t y_pred[N_DOFS];
    topographic_cnn2d_infer(GOLDEN_X_FLAT, y_pred);

    double max_err = 0.0;
    std::cout << "=== Vitis HLS C-Simulation: TopographicCNN2D (5-DoF) ===\n";
    for (int d = 0; d < N_DOFS; ++d) {
        double pred_f = static_cast<double>(y_pred[d]);
        double ref_f = static_cast<double>(GOLDEN_Y_REF[d]);
        double err = std::abs(pred_f - ref_f);
        if (err > max_err) {
            max_err = err;
        }
        std::cout << "DoF " << d << ": HLS=" << pred_f
                  << " deg | Ref=" << ref_f
                  << " deg | |err|=" << err << " deg\n";
    }

    std::cout << "Max absolute error: " << max_err << " deg\n";
    if (max_err > 1.0) {
        std::cerr << "FAIL: erro acima de 1.0 grau!\n";
        return 1;
    }
    std::cout << "PASS: C-Simulation validada!\n";
    return 0;
}
