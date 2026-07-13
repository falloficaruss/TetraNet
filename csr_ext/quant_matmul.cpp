#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>

// ─────────────────────────────────────────────────────────────────
//  1. Standard INT32 matmul (baseline — uses MULT + ADD)
// ─────────────────────────────────────────────────────────────────
at::Tensor int32_matmul(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dtype() == at::kInt, "A must be INT32");
    TORCH_CHECK(B.dtype() == at::kInt, "B must be INT32");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "dim mismatch");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, at::kInt);
    const int32_t* a_ptr = A.data_ptr<int32_t>();
    const int32_t* b_ptr = B.data_ptr<int32_t>();
    int32_t* c_ptr = C.data_ptr<int32_t>();

    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += a_ptr[m * K + k] * b_ptr[k * N + n];
            }
            c_ptr[m * N + n] = sum;
        }
    }
    return C;
}

// ─────────────────────────────────────────────────────────────────
//  2. Ternary matmul — weights in {-1, 0, 1}, no MULT
// ─────────────────────────────────────────────────────────────────
at::Tensor ternary_matmul(at::Tensor A, at::Tensor W) {
    TORCH_CHECK(A.dtype() == at::kInt, "A must be INT32");
    TORCH_CHECK(W.dtype() == at::kChar, "W must be INT8");
    TORCH_CHECK(A.dim() == 2 && W.dim() == 2, "must be 2D");
    TORCH_CHECK(A.size(1) == W.size(0), "dim mismatch");

    int M = A.size(0);
    int K = A.size(1);
    int N = W.size(1);

    auto C = torch::zeros({M, N}, at::kInt);
    const int32_t* a_ptr = A.data_ptr<int32_t>();
    const int8_t*  w_ptr = W.data_ptr<int8_t>();
    int32_t* c_ptr = C.data_ptr<int32_t>();

    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                int32_t a = a_ptr[m * K + k];
                int8_t  w = w_ptr[k * N + n];
                // branchless: w ∈ {-1,0,1} → conditional negate/zero via arithmetic
                int32_t mask_pos = (w > 0);
                int32_t mask_neg = (w < 0);
                sum += mask_pos * a - mask_neg * a;
            }
            c_ptr[m * N + n] = sum;
        }
    }
    return C;
}

// ─────────────────────────────────────────────────────────────────
//  3. Quaternary shift matmul — packed 2-bit weights
//     00 → +1, 01 → -1, 10 → +c (>>shift_bits), 11 → -c
// ─────────────────────────────────────────────────────────────────
at::Tensor shift_matmul(at::Tensor A, at::Tensor W_packed, int64_t shift_bits) {
    TORCH_CHECK(A.dtype() == at::kInt, "A must be INT32");
    TORCH_CHECK(W_packed.dtype() == at::kByte, "W_packed must be UINT8");
    TORCH_CHECK(A.dim() == 2 && W_packed.dim() == 2, "must be 2D");

    int M = A.size(0);
    int K = A.size(1);
    int N = W_packed.size(1);
    int K_packed = W_packed.size(0);
    int K_unpacked = K_packed * 4;

    TORCH_CHECK(K_unpacked >= K, "packed dim too small");
    TORCH_CHECK(K_unpacked - K < 4, "packed dim mismatch");

    auto C = torch::zeros({M, N}, at::kInt);
    const int32_t* a_ptr = A.data_ptr<int32_t>();
    const uint8_t* w_ptr = W_packed.data_ptr<uint8_t>();
    int32_t* c_ptr = C.data_ptr<int32_t>();

    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int pk = 0; pk < K_packed; pk++) {
                uint8_t packed = w_ptr[pk * N + n];
                for (int b = 0; b < 4; b++) {
                    int k = pk * 4 + b;
                    if (k >= K) break;
                    int32_t a = a_ptr[m * K + k];
                    uint8_t code = (packed >> (b * 2)) & 0x3;
                    int32_t shifted = (code & 2) ? (a >> shift_bits) : a;
                    int32_t term = (code & 1) ? -shifted : shifted;
                    sum += term;
                }
            }
            c_ptr[m * N + n] = sum;
        }
    }
    return C;
}

// ─────────────────────────────────────────────────────────────────
//  4. Uniform 2-bit matmul — packed codes + level table (uses MUL)
//     codes: 0=-1, 1=-1/3, 2=+1/3, 3=+1
// ─────────────────────────────────────────────────────────────────
at::Tensor uniform_matmul(at::Tensor A, at::Tensor W_packed, int64_t K) {
    TORCH_CHECK(A.dtype() == at::kInt, "A must be INT32");
    TORCH_CHECK(W_packed.dtype() == at::kByte, "W_packed must be UINT8");
    TORCH_CHECK(A.dim() == 2 && W_packed.dim() == 2, "must be 2D");
    TORCH_CHECK(A.size(1) == K, "K mismatch");

    int M = A.size(0);
    int N = W_packed.size(1);
    int K_packed = W_packed.size(0);

    auto C = torch::zeros({M, N}, at::kFloat);
    const int32_t* a_ptr = A.data_ptr<int32_t>();
    const uint8_t* w_ptr = W_packed.data_ptr<uint8_t>();
    float* c_ptr = C.data_ptr<float>();

    const float levels[4] = {-1.0f, -1.0f / 3.0f, 1.0f / 3.0f, 1.0f};

    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int pk = 0; pk < K_packed; pk++) {
                uint8_t packed = w_ptr[pk * N + n];
                for (int b = 0; b < 4; b++) {
                    int k = pk * 4 + b;
                    if (k >= K) break;
                    float a = static_cast<float>(a_ptr[m * K + k]);
                    uint8_t code = (packed >> (b * 2)) & 0x3;
                    sum += a * levels[code];
                }
            }
            c_ptr[m * N + n] = sum;
        }
    }
    return C;
}

// ─────────────────────────────────────────────────────────────────
//  Pack utilities
// ─────────────────────────────────────────────────────────────────
at::Tensor pack_quaternary_weights(at::Tensor weight, double c) {
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be FP32");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");

    int K = weight.size(0);
    int N = weight.size(1);
    int K_packed = (K + 3) / 4;

    float c_float = static_cast<float>(c);
    float boundary = (1.0f + c_float) / 2.0f;

    auto packed = torch::zeros({K_packed, N}, at::kByte);
    uint8_t* packed_ptr = packed.data_ptr<uint8_t>();
    const float* w_ptr = weight.data_ptr<float>();

    for (int k = 0; k < K; k++) {
        int pk = k / 4;
        int bit_offset = (k % 4) * 2;
        for (int n = 0; n < N; n++) {
            float val = w_ptr[k * N + n];
            uint8_t code;
            if (val <= -boundary)       code = 1;
            else if (val <= 0.0f)       code = 3;
            else if (val <= boundary)   code = 2;
            else                        code = 0;
            packed_ptr[pk * N + n] |= (code << bit_offset);
        }
    }
    return packed;
}

at::Tensor pack_ternary_weights(at::Tensor weight) {
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be FP32");

    auto W_int8 = torch::zeros_like(weight, at::kChar);
    int8_t* w_ptr = W_int8.data_ptr<int8_t>();
    const float* src = weight.data_ptr<float>();

    int64_t numel = weight.numel();
    for (int64_t i = 0; i < numel; i++) {
        float v = src[i];
        if (v > 0.5f)       w_ptr[i] = 1;
        else if (v < -0.5f) w_ptr[i] = -1;
        else                w_ptr[i] = 0;
    }
    return W_int8;
}

at::Tensor pack_uniform_weights(at::Tensor weight) {
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be FP32");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");

    int K = weight.size(0);
    int N = weight.size(1);
    int K_packed = (K + 3) / 4;
    const float levels[4] = {-1.0f, -1.0f / 3.0f, 1.0f / 3.0f, 1.0f};

    auto packed = torch::zeros({K_packed, N}, at::kByte);
    uint8_t* packed_ptr = packed.data_ptr<uint8_t>();
    const float* w_ptr = weight.data_ptr<float>();

    for (int k = 0; k < K; k++) {
        int pk = k / 4;
        int bit_offset = (k % 4) * 2;
        for (int n = 0; n < N; n++) {
            float val = w_ptr[k * N + n];
            int best = 0;
            float best_d = fabsf(val - levels[0]);
            for (int i = 1; i < 4; i++) {
                float d = fabsf(val - levels[i]);
                if (d < best_d) { best_d = d; best = i; }
            }
            packed_ptr[pk * N + n] |= (static_cast<uint8_t>(best) << bit_offset);
        }
    }
    return packed;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int32_matmul", &int32_matmul, "INT32 matmul (MULT+ADD baseline)");
    m.def("ternary_matmul", &ternary_matmul, "Ternary matmul (MUX+ADD)");
    m.def("shift_matmul", &shift_matmul, "Quaternary shift matmul");
    m.def("uniform_matmul", &uniform_matmul, "Uniform 2-bit LUT matmul");
    m.def("pack_quaternary_weights", &pack_quaternary_weights, "Pack FP32 → 2-bit quaternary");
    m.def("pack_ternary_weights", &pack_ternary_weights, "Pack FP32 → INT8 ternary");
    m.def("pack_uniform_weights", &pack_uniform_weights, "Pack FP32 → 2-bit uniform");
}
