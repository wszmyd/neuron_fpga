#include "xparameters.h"
#include "xil_io.h"
#include "xil_printf.h"
#include <stdint.h>
#include "perceptron_ref_model.h"


#ifndef NEURON_BASEADDR
  #if defined(XPAR_NEURON_AXI_LITE_0_S00_AXI_BASEADDR)
    #define NEURON_BASEADDR XPAR_NEURON_AXI_LITE_0_S00_AXI_BASEADDR
  #elif defined(XPAR_NEURON_AXI_LITE_0_BASEADDR)
    #define NEURON_BASEADDR XPAR_NEURON_AXI_LITE_0_BASEADDR
  #else
    #define NEURON_BASEADDR 0x43C00000U
  #endif
#endif

#define NEURON_REG_CTRL   0x00U
#define NEURON_REG_INPUT  0x04U
#define NEURON_REG_W0     0x08U
#define NEURON_REG_W1     0x0CU
#define NEURON_REG_W2     0x10U
#define NEURON_REG_WB     0x14U
#define NEURON_REG_RW0    0x18U
#define NEURON_REG_RW1    0x1CU
#define NEURON_REG_RW2    0x20U
#define NEURON_REG_RWB    0x24U

#define CTRL_START        0x00000001U
#define CTRL_MODE_TRAIN   0x00000002U
#define CTRL_INIT         0x00000004U
#define CTRL_CLEAR_DONE   0x00000008U
#define CTRL_DONE         0x00000008U
#define CTRL_READY        0x00000010U
#define CTRL_RESULT       0x00000100U

#define FRAC_BITS         12
#define DEFAULT_EPOCHS    300U
#define MAX_EPOCHS        5000U

static uint32_t g_epochs = DEFAULT_EPOCHS;
static uint32_t g_rng = 0x12345678U;

static inline void neuron_write(uint32_t offset, uint32_t value) {
    Xil_Out32(NEURON_BASEADDR + offset, value);
}

static inline uint32_t neuron_read(uint32_t offset) {
    return Xil_In32(NEURON_BASEADDR + offset);
}

static uint32_t pack20(int32_t value) {
    return ((uint32_t)value) & 0x000FFFFFU;
}

static int32_t sign_extend20(uint32_t value) {
    value &= 0x000FFFFFU;
    if (value & 0x00080000U) {
        value |= 0xFFF00000U;
    }
    return (int32_t)value;
}

static char uart_get_char(void) {
    return (char)inbyte();
}

static void uart_get_line(char *buf, uint32_t max_len) {
    uint32_t i = 0U;
    while (i + 1U < max_len) {
        char c = uart_get_char();

        if (c == '\r' || c == '\n') {
            xil_printf("\r\n");
            break;
        }

        if ((c == 8 || c == 127) && i > 0U) {
            i--;
            xil_printf("\b \b");
            continue;
        }

        if (c >= 32 && c <= 126) {
            buf[i++] = c;
            xil_printf("%c", c);
        }
    }
    buf[i] = '\0';
}

static int32_t parse_int(const char *s) {
    int32_t sign = 1;
    int32_t val = 0;

    while (*s == ' ' || *s == '\t') {
        s++;
    }

    if (*s == '-') {
        sign = -1;
        s++;
    } else if (*s == '+') {
        s++;
    }

    while (*s >= '0' && *s <= '9') {
        val = (val * 10) + (*s - '0');
        s++;
    }

    return sign * val;
}

static uint32_t read_uint_prompt(const char *prompt, uint32_t min_value, uint32_t max_value) {
    char line[32];
    uint32_t value;

    while (1) {
        xil_printf("%s", prompt);
        uart_get_line(line, sizeof(line));
        value = (uint32_t)parse_int(line);

        if (value >= min_value && value <= max_value) {
            return value;
        }

        xil_printf("Invalid value. Range: %d..%d\r\n",
                   (int)min_value, (int)max_value);
    }
}

static int32_t read_int_prompt(const char *prompt, int32_t min_value, int32_t max_value) {
    char line[32];
    int32_t value;

    while (1) {
        xil_printf("%s", prompt);
        uart_get_line(line, sizeof(line));
        value = parse_int(line);

        if (value >= min_value && value <= max_value) {
            return value;
        }

        xil_printf("Invalid value. Range: %d..%d\r\n",
                   (int)min_value, (int)max_value);
    }
}

static void neuron_init_weights(int32_t w0, int32_t w1, int32_t w2, int32_t wbias) {
    neuron_write(NEURON_REG_W0, pack20(w0));
    neuron_write(NEURON_REG_W1, pack20(w1));
    neuron_write(NEURON_REG_W2, pack20(w2));
    neuron_write(NEURON_REG_WB, pack20(wbias));
    neuron_write(NEURON_REG_CTRL, CTRL_INIT);
}

static uint8_t neuron_run_sample(uint8_t x, uint8_t expected, uint8_t training_mode) {
    uint32_t ctrl_mode = training_mode ? CTRL_MODE_TRAIN : 0U;
    uint32_t ctrl;
    uint32_t timeout;

    neuron_write(NEURON_REG_CTRL, ctrl_mode | CTRL_CLEAR_DONE);
    neuron_write(NEURON_REG_INPUT, ((uint32_t)(expected & 1U) << 8) | (uint32_t)(x & 7U));
    neuron_write(NEURON_REG_CTRL, ctrl_mode | CTRL_START);

    timeout = 1000000U;
    do {
        ctrl = neuron_read(NEURON_REG_CTRL);
        timeout--;
    } while (((ctrl & CTRL_DONE) == 0U) && (timeout != 0U));

    if (timeout == 0U) {
        xil_printf("ERROR: timeout for x=%d\r\n", x);
        return 0xFFU;
    }

    return (uint8_t)((ctrl & CTRL_RESULT) ? 1U : 0U);
}


static void read_weights(int32_t *w0, int32_t *w1, int32_t *w2, int32_t *wb) {
    *w0 = sign_extend20(neuron_read(NEURON_REG_RW0));
    *w1 = sign_extend20(neuron_read(NEURON_REG_RW1));
    *w2 = sign_extend20(neuron_read(NEURON_REG_RW2));
    *wb = sign_extend20(neuron_read(NEURON_REG_RWB));
}

static perceptron_weights_t read_weights_struct(void) {
    perceptron_weights_t weights;

    read_weights(&weights.w0, &weights.w1, &weights.w2, &weights.bias);

    return weights;
}

static void print_weights(void) {
    int32_t w0, w1, w2, wb;
    read_weights(&w0, &w1, &w2, &wb);

    xil_printf("Current weights, raw Q7.12:\r\n");
    xil_printf("  w0=%d  w1=%d  w2=%d  bias=%d\r\n", w0, w1, w2, wb);

    xil_printf("As real values, with FRAC_BITS=%d:\r\n", FRAC_BITS);
    xil_printf("  w0=%d/%d  w1=%d/%d  w2=%d/%d  bias=%d/%d\r\n",
               w0, 1 << FRAC_BITS, w1, 1 << FRAC_BITS,
               w2, 1 << FRAC_BITS, wb, 1 << FRAC_BITS);

    xil_printf("Line for weights.mem, order {bias,w2,w1,w0}:\r\n");
    xil_printf("  %05x%05x%05x%05x\r\n",
               (unsigned int)pack20(wb),
               (unsigned int)pack20(w2),
               (unsigned int)pack20(w1),
               (unsigned int)pack20(w0));
}

static uint32_t test_all_vectors(uint8_t verbose) {
    uint32_t errors = 0U;
    perceptron_weights_t weights = read_weights_struct();

    if (verbose) {
        xil_printf("\r\n");
        xil_printf("========== FPGA vs C REFERENCE MODEL TEST ==========\r\n");
        xil_printf("Current FPGA weights, raw Q7.12:\r\n");
        xil_printf("  w0=%d  w1=%d  w2=%d  bias=%d\r\n",
                   weights.w0, weights.w1, weights.w2, weights.bias);
        xil_printf("\r\n");

        xil_printf("%-5s %-5s %8s %6s %6s %-10s %-11s\r\n",
                   "x", "AND", "c_sum", "C_ref", "FPGA", "FPGA_vs_C", "FPGA_vs_AND");
        xil_printf("-------------------------------------------------------------\r\n");
    }

    for (uint8_t x = 0; x < 8; x++) {
        uint8_t expected = ref_expected_and3(x);
        uint8_t fpga_result = neuron_run_sample(x, expected, 0U);
        uint8_t c_reference = ref_perceptron_predict(x, &weights);
        int32_t c_sum = ref_perceptron_sum(x, &weights);

        uint8_t fpga_vs_c_ok = (fpga_result == c_reference);
        uint8_t fpga_vs_and_ok = (fpga_result == expected);

        if (verbose) {
            char x_text[4];
            x_text[0] = ((x >> 2) & 1U) ? '1' : '0';
            x_text[1] = ((x >> 1) & 1U) ? '1' : '0';
            x_text[2] = (x & 1U) ? '1' : '0';
            x_text[3] = '\0';

            xil_printf("%-5s %-5d %8d %6d %6d %-10s %-11s\r\n",
                       x_text,
                       expected,
                       c_sum,
                       c_reference,
                       fpga_result,
                       fpga_vs_c_ok ? "OK" : "ERROR",
                       fpga_vs_and_ok ? "OK" : "ERROR");
        }

        if (!fpga_vs_c_ok || !fpga_vs_and_ok) {
            errors++;
        }
    }

    if (verbose) {
        xil_printf("-------------------------------------------------------------\r\n");
        xil_printf("Verification errors: %d/8\r\n", (int)errors);
        xil_printf("=============================================================\r\n");
    }

    return errors;
}

static void train_fixed_epochs(uint32_t epochs, uint8_t verbose) {
    uint32_t epoch;

    for (epoch = 0U; epoch < epochs; epoch++) {
        uint32_t train_errors = 0U;

        for (uint8_t x = 0; x < 8; x++) {
            uint8_t exp = ref_expected_and3(x);
            uint8_t y = neuron_run_sample(x, exp, 1U);
            if (y != exp) {
                train_errors++;
            }
        }

        if (verbose && ((epoch % 25U) == 0U || train_errors == 0U)) {
            xil_printf("  epoch=%d train_errors=%d\r\n", (int)epoch, (int)train_errors);
        }
    }
}

static void train_until_success(uint32_t max_epochs) {
    uint32_t epoch;

    for (epoch = 0U; epoch < max_epochs; epoch++) {
        train_fixed_epochs(1U, 0U);

        uint32_t errors = test_all_vectors(0U);
        if ((epoch % 25U) == 0U || errors == 0U) {
            xil_printf("  epoch=%d verification_errors=%d\r\n", (int)(epoch + 1U), (int)errors);
        }

        if (errors == 0U) {
            xil_printf("Success: learned after %d epochs.\r\n", (int)(epoch + 1U));
            return;
        }
    }

    xil_printf("Full correctness was not reached after %d epochs.\r\n", (int)max_epochs);
}

static uint32_t lcg_rand(void) {
    g_rng = 1664525U * g_rng + 1013904223U;
    return g_rng;
}

static int32_t random_weight_q12(void) {
    /*
     * Generates a random weight in the range of about -0.25..+0.25 in Q7.12.
     * 0.25 * 4096 = 1024.
     */
    int32_t r = (int32_t)(lcg_rand() % 2049U) - 1024;
    return r;
}

static void force_manual_weights(void) {
    xil_printf("Enter weights in raw Q7.12 format.\r\n");
    xil_printf("Example: 4096 means 1.0, -2048 means -0.5.\r\n");

    int32_t w0 = read_int_prompt("w0 = ", -524288, 524287);
    int32_t w1 = read_int_prompt("w1 = ", -524288, 524287);
    int32_t w2 = read_int_prompt("w2 = ", -524288, 524287);
    int32_t wb = read_int_prompt("bias = ", -524288, 524287);

    neuron_init_weights(w0, w1, w2, wb);
    xil_printf("Weights forced manually.\r\n");
    print_weights();
}

static void classify_one_vector(void) {
    uint32_t x = read_uint_prompt("Enter x as a number from 0 to 7: ", 0U, 7U);
    perceptron_weights_t weights = read_weights_struct();

    uint8_t expected = ref_expected_and3((uint8_t)x);
    uint8_t fpga_result = neuron_run_sample((uint8_t)x, expected, 0U);
    uint8_t c_reference = ref_perceptron_predict((uint8_t)x, &weights);
    int32_t c_sum = ref_perceptron_sum((uint8_t)x, &weights);

    xil_printf("Result for x=%d%d%d: expected_AND=%d c_sum=%d c_ref=%d fpga=%d %s\r\n",
               (int)((x >> 2) & 1U), (int)((x >> 1) & 1U), (int)(x & 1U),
               (int)expected, (int)c_sum, (int)c_reference, (int)fpga_result,
               (fpga_result == c_reference) ? "FPGA=C_OK" : "FPGA=C_ERROR");
}

static void show_menu(void) {
    xil_printf("\r\n");
    xil_printf("===== FPGA PERCEPTRON MENU =====\r\n");
    xil_printf("1 - Show current weights\r\n");
    xil_printf("2 - Manually force initial weights\r\n");
    xil_printf("3 - Set number of training epochs, currently %d\r\n", (int)g_epochs);
    xil_printf("4 - Train for the selected number of epochs\r\n");
    xil_printf("5 - Train until success, early stopping\r\n");
    xil_printf("6 - Test all 8 AND input vectors and compare with C reference model\r\n");
    xil_printf("7 - Check a single input vector\r\n");
    xil_printf("8 - Randomize initial weights\r\n");
    xil_printf("9 - Restore default weights 0,0,0,-10\r\n");
    xil_printf("h - Help: register map and weight format\r\n");
    xil_printf("Select option: ");
}

static void print_help(void) {
    xil_printf("\r\nExtension description:\r\n");
    xil_printf("- The program communicates with the IP core through AXI4-Lite.\r\n");
    xil_printf("- Weights are signed 20-bit Q7.12 values. Therefore, 1.0 = 4096.\r\n");
    xil_printf("- CTRL bit0=start, bit1=mode, bit2=init, bit3=clear/done, bit8=result.\r\n");
    xil_printf("- INPUT bits[2:0]=input vector x, bit8=expected_result.\r\n");
    xil_printf("- Training in SDK repeatedly sends samples to the IP core with mode=1.\r\n");
    xil_printf("- Early stopping ends training when the 8-vector test has no errors.\r\n");
    xil_printf("- A separate C reference model calculates the same perceptron equation as the FPGA IP.\r\n");
}

int main(void) {
    xil_printf("\r\n--- FPGA/Zynq perceptron application ---\r\n");
    xil_printf("NEURON_BASEADDR = 0x%x\r\n", (unsigned int)NEURON_BASEADDR);

    neuron_init_weights(0, 0, 0, -10);
    xil_printf("Default weights loaded.\r\n");
    print_weights();

    while (1) {
        show_menu();
        char cmd = uart_get_char();
        xil_printf("%c\r\n", cmd);

        switch (cmd) {
        case '1':
            print_weights();
            break;

        case '2':
            force_manual_weights();
            break;

        case '3':
            g_epochs = read_uint_prompt("Number of epochs 1..5000 = ", 1U, MAX_EPOCHS);
            xil_printf("Epoch count set to %d\r\n", (int)g_epochs);
            break;

        case '4':
            xil_printf("Training for %d epochs...\r\n", (int)g_epochs);
            train_fixed_epochs(g_epochs, 1U);
            xil_printf("Training finished.\r\n");
            print_weights();
            test_all_vectors(1U);
            break;

        case '5':
            train_until_success(g_epochs);
            print_weights();
            test_all_vectors(1U);
            break;

        case '6':
            test_all_vectors(1U);
            break;

        case '7':
            classify_one_vector();
            break;

        case '8': {
            int32_t w0 = random_weight_q12();
            int32_t w1 = random_weight_q12();
            int32_t w2 = random_weight_q12();
            int32_t wb = random_weight_q12();
            neuron_init_weights(w0, w1, w2, wb);
            xil_printf("Initial weights randomized.\r\n");
            print_weights();
            break;
        }

        case '9':
            neuron_init_weights(0, 0, 0, -10);
            xil_printf("Default weights restored.\r\n");
            print_weights();
            break;

        case 'h':
        case 'H':
            print_help();
            break;

        default:
            xil_printf("Unknown option.\r\n");
            break;
        }
    }

    return 0;
}
