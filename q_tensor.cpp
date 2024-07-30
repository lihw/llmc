
#include "q_tensor.h"

#include <spdlog/spdlog.h>

#include <vector>

#include <cassert>

namespace q {

Parameters g_params{};

std::vector<Tensor*> g_recycles;

void Tensor::quantize(const float* x) {
    const int GS = (int)g_params.GS;

    const int n = dim[0] * dim[1] * dim[2];

    // int8
    int numGroups = n / GS;
    float Q_MAX = 127.0f;

    if (values == nullptr ) {
        values = new int8_t[n];
        scales = new float[numGroups];

        g_recycles.push_back(this);
    }

    for (int group = 0; group < numGroups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        scales[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quantValue = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t)round(quantValue); // round and clamp
            values[group * GS + i] = quantized;
        }
    }
}

void Tensor::dequantize(float* x) const {
    const int GS = (int)g_params.GS;

    const int n = dim[0] * dim[1] * dim[2];

    // int8
    for (int i = 0; i < n; i++) {
        x[i] = values[i] * scales[i / GS];
    }
}

void Tensor::matmul(float* out, Tensor* x) {
    const int GS = (int)g_params.GS;

    // this
    // W (d,n) x (n,) -> out (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    assert(dim[1] == x->dim[0]); // The x is a 1d vector 

    const int d = (int)dim[0];
    const int n = (int)dim[1];

    const int8_t* xx = &reinterpret_cast<Tensor*>(x)->values[0];
    const int8_t* ww = &values[0];

    const float* xs = &reinterpret_cast<Tensor*>(x)->scales[0];
    const float* ws = &scales[0];

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) xx[j + k]) * ((int32_t) ww[in + j + k]);
            }
            val += ((float) ival) * ws[(in + j) / GS] * xs[j / GS];
            ival = 0;
        }

        out[i] = val;
    }
}
    
void Tensor::upload(void** x) {
    int8_t* p = (int8_t*)*x;
    values = p;
    p += dim[0] * dim[1] * dim[2];
    float* q = (float*)p;
    scales = q;
    q += dim[0] * dim[1] * dim[2] / g_params.GS;
    *x = q;
}

Tensor* create(Tensor::Backend backend, Tensor::DataType dataType,
        int rows, int cols, int channels) {

    const int GS = (int)g_params.GS;

    assert(rows > 0 && cols > 0 && channels > 0);

    // FIXME: what if dim is invalid

    Tensor* tensor;
    if (backend == Tensor::Backend::VANILLA) {
        tensor = new Tensor;
    }

    tensor->dim[0] = rows;
    tensor->dim[1] = cols;
    tensor->dim[2] = channels;

    tensor->values = nullptr;
    tensor->scales = nullptr;
    
    return tensor;
}

void initialize(const Parameters& params) {
    g_params = params;

    g_recycles.reserve(1024);
}

void finalize() {
    for (auto& t : g_recycles) {
        auto* tt = reinterpret_cast<Tensor*>(t);
        delete [] tt->scales;
        delete[] tt->values;
    }
}

}; // namespace q
