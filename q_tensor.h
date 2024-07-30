
#ifndef Q_TENSOR_H
#define Q_TENSOR_H

#include <fmt/core.h>

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

extern int GS;

namespace q {

struct Tensor {
    int8_t* q;    // quantized values
    float* s; // scaling factors
    int rows; // (rows, cols) if 2D vector/matrix, (rows,) if 1D vector
    int cols;
};

void matmul(float* xout, Tensor *x, Tensor *w) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized
    //
    assert(w->cols == x->rows);

    int d = w->rows;
    int n = w->cols;

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
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
Tensor *init_quantized_tensors(void **ptr, int n, int rows, int cols) {
    void *p = *ptr;
    Tensor *res = (Tensor*)malloc(n * sizeof(Tensor));
    int size_each = rows * cols;
    for(int i = 0; i < n; i++) {
        res[i].rows = rows;
        res[i].cols = cols;
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(Tensor *qx, float* x) {
    int n = qx->rows * qx->cols;
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(Tensor *qx, float* x) {

    int n = qx->rows * qx->cols;
 
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

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
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

void dumpVector(float* x, size_t dim, const char* name) {
#define out stdout
    auto t = fmt::format("{}: [{}, {}]", name, dim, 1);
    for (int j = 0; j < 5; j++) {
        t += fmt::format("{:5.3f}, ", x[j]);
    }
    t += fmt::format("{:5.3f}, ", x[dim - 1]);
    fprintf(out, "%s\n", t.c_str());
#undef out
}

}; // namespace q

#endif

