`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08.06.2026 21:01:50
// Design Name: 
// Module Name: neuron
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module neuron #(
    parameter int WEIGHT_WIDTH  = 20,
    parameter logic signed [WEIGHT_WIDTH-1:0] LEARNING_RATE = 20'sd512  // 0.125
)(
    input  logic clk,
    input  logic rstn,

    input  logic [2:0] in_data,
    input  logic       in_data_vld,
    output logic       in_data_rdy,

    input  logic       expected_result_data,
    input  logic       mode,                 // 0 - work, 1 - learning

    input  logic [4*WEIGHT_WIDTH-1:0] init_weights_data,
    input  logic                      init_weights_vld,

    output logic       result_data,
    output logic       result_vld,
    output logic [4*WEIGHT_WIDTH-1:0] result_weights
);

    logic signed [WEIGHT_WIDTH-1:0] w0, w1, w2, wbias;
    logic signed [WEIGHT_WIDTH-1:0] w0_next, w1_next, w2_next, wbias_next;

    logic signed [WEIGHT_WIDTH+2:0] sum;
    logic prediction;

    function automatic logic signed [WEIGHT_WIDTH+2:0] sx_weight(
        input logic signed [WEIGHT_WIDTH-1:0] value
    );
        sx_weight = {{3{value[WEIGHT_WIDTH-1]}}, value};
    endfunction

    assign in_data_rdy = 1'b1;

    assign result_weights = {wbias, w2, w1, w0};

    always_comb begin
        sum = sx_weight(wbias);
        if (in_data[0]) sum = sum + sx_weight(w0);
        if (in_data[1]) sum = sum + sx_weight(w1);
        if (in_data[2]) sum = sum + sx_weight(w2);
    end

    localparam logic signed [WEIGHT_WIDTH+2:0] SUM_ZERO = '0;
    assign prediction = (sum >= SUM_ZERO);

    always_comb begin
        w0_next    = w0;
        w1_next    = w1;
        w2_next    = w2;
        wbias_next = wbias;

        if (mode && in_data_vld && in_data_rdy) begin
            if (expected_result_data && !prediction) begin
                if (in_data[0]) w0_next = w0 + LEARNING_RATE;
                if (in_data[1]) w1_next = w1 + LEARNING_RATE;
                if (in_data[2]) w2_next = w2 + LEARNING_RATE;
                wbias_next = wbias + LEARNING_RATE;
            end else if (!expected_result_data && prediction) begin
                if (in_data[0]) w0_next = w0 - LEARNING_RATE;
                if (in_data[1]) w1_next = w1 - LEARNING_RATE;
                if (in_data[2]) w2_next = w2 - LEARNING_RATE;
                wbias_next = wbias - LEARNING_RATE;
            end
        end
    end

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            w0          <= '0;
            w1          <= '0;
            w2          <= '0;
            wbias       <= '0;
            result_data <= 1'b0;
            result_vld  <= 1'b0;
        end else begin
            result_vld <= 1'b0;

            if (init_weights_vld) begin
                w0          <= init_weights_data[0*WEIGHT_WIDTH +: WEIGHT_WIDTH];
                w1          <= init_weights_data[1*WEIGHT_WIDTH +: WEIGHT_WIDTH];
                w2          <= init_weights_data[2*WEIGHT_WIDTH +: WEIGHT_WIDTH];
                wbias       <= init_weights_data[3*WEIGHT_WIDTH +: WEIGHT_WIDTH];
                result_data <= 1'b0;
                result_vld  <= 1'b0;
            end else if (in_data_vld && in_data_rdy) begin
                result_data <= prediction;
                result_vld  <= 1'b1;
                w0          <= w0_next;
                w1          <= w1_next;
                w2          <= w2_next;
                wbias       <= wbias_next;
            end
        end
    end

endmodule

