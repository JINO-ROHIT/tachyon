#include <cmath>

#include "include/cpu/sampler.h"

int sample_greedy(const Tensor<1>& logits, float temperature) {
    int n = logits.shape[0];

    int best_idx = 0;
    float best_score = logits[0] / temperature;

    for (int i = 1; i < n; i++) {
        float score = logits[i] / temperature;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return best_idx;
}


int temperature_sampling(const Tensor<1>& logits, float temperature) {
    int n = logits.shape[0];

    float* scores = new float[n];

    // Scale logits by temperature
    float max_score = -INFINITY; // keep track of this for softmax
    for (int i = 0; i < n; i++) {
        scores[i] = logits[i] / temperature;
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }

    for (int i = 0; i < n; i++) {
        scores[i] /= sum_exp;
    }

    float r = (float)rand() / RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += scores[i];
        if (r <= cdf) {
            delete[] scores;
            return i; 
        }
    }

    delete[] scores;
    return n - 1;
}