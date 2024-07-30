
#ifndef Q_NEURAL_H
#define Q_NERUAL_H

#include <ctype.h>
#include <stdint.h>

namespace q {

void rope(float* q, float* k, int pos, int dim, int kv_dim, int head_size) {
    for (int i = 0; i < dim; i+=2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? q : k; // the vector to rotate (query or key)
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// out = dot(a, b)
float dot(const float* a, const float* b, size_t start, size_t end) {
    float sum  = 0;

    for (auto i = start; i < end; i += 4) {
        sum += a[i] * b[i];
        sum += a[i + 1] * b[i + 1];
        sum += a[i + 2] * b[i + 2];
        sum += a[i + 3] * b[i + 3];
    }

    return sum;
}

// out += x * scale
void scale1(float* out, const float* x, float scale, size_t start, size_t end) {
    for (auto i = start; i < end; i += 4) {
        out[i]     += x[i] * scale;
        out[i + 1] += x[i + 1] * scale;
        out[i + 2] += x[i + 2] * scale;
        out[i + 3] += x[i + 3] * scale;
    }
}

void add1(float* out, const float* x, size_t dim) {
    for (size_t i = 0; i < dim; i += 4) {
        out[i]     += x[i];
        out[i + 1] += x[i + 1];
        out[i + 2] += x[i + 2];
        out[i + 3] += x[i + 3];
    }
}

void swiglu(float* out, const float* w, size_t dim) {
    for (int i = 0; i < dim; i += 4) {
        auto scale = [](float val, float scale) {
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= scale;
            return val;
        };
        out[i]     = scale(out[i], w[i]);
        out[i + 1] = scale(out[i + 1], w[i + 1]);
        out[i + 2] = scale(out[i + 2], w[i + 2]);
        out[i + 3] = scale(out[i + 3], w[i + 3]);
    }
}

}; // namespace q

#endif // !Q_NERUAL_H
