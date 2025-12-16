#include "tensor.h"

int sample_greedy(const Tensor<1>& logits, float temperature);
int temperature_sampling(const Tensor<1>& logits, float temperature);