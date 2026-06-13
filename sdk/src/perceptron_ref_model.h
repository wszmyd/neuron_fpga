#ifndef PERCEPTRON_REF_MODEL_H
#define PERCEPTRON_REF_MODEL_H

#include <stdint.h>

/*
 * Software reference model for the 3-input perceptron implemented in FPGA.
 *
 * Weight format:
 *   - signed raw Q7.12 values,
 *   - the same format as used by the FPGA IP,
 *   - 1.0 = 4096.
 *
 * Input bit mapping:
 *   x0 = bit 0
 *   x1 = bit 1
 *   x2 = bit 2
 */

typedef struct {
    int32_t w0;
    int32_t w1;
    int32_t w2;
    int32_t bias;
} perceptron_weights_t;

/* Truth table reference for a 3-input AND gate. */
uint8_t ref_expected_and3(uint8_t x);

/* Reference perceptron inference model:
 * sum = bias + x0*w0 + x1*w1 + x2*w2
 * y   = 1 when sum >= 0, otherwise 0
 */
uint8_t ref_perceptron_predict(uint8_t x, const perceptron_weights_t *weights);

/* Optional helper: calculates raw perceptron sum for debugging. */
int32_t ref_perceptron_sum(uint8_t x, const perceptron_weights_t *weights);

/* Optional helper: compares FPGA output with the C reference model. */
uint8_t ref_compare_fpga_output(uint8_t x,
                                uint8_t fpga_output,
                                const perceptron_weights_t *weights);

#endif /* PERCEPTRON_REF_MODEL_H */
