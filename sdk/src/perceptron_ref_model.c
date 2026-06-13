#include "perceptron_ref_model.h"

uint8_t ref_expected_and3(uint8_t x) {
    return ((x & 0x7U) == 0x7U) ? 1U : 0U;
}

int32_t ref_perceptron_sum(uint8_t x, const perceptron_weights_t *weights) {
    int32_t sum = weights->bias;

    if (x & 0x1U) {
        sum += weights->w0;
    }

    if (x & 0x2U) {
        sum += weights->w1;
    }

    if (x & 0x4U) {
        sum += weights->w2;
    }

    return sum;
}

uint8_t ref_perceptron_predict(uint8_t x, const perceptron_weights_t *weights) {
    int32_t sum = ref_perceptron_sum(x, weights);
    return (sum >= 0) ? 1U : 0U;
}

uint8_t ref_compare_fpga_output(uint8_t x,
                                uint8_t fpga_output,
                                const perceptron_weights_t *weights) {
    uint8_t c_reference = ref_perceptron_predict(x, weights);
    return (fpga_output == c_reference) ? 1U : 0U;
}
