`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08.06.2026 21:02:53
// Design Name: 
// Module Name: tb_neuron
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


module tb_neuron;
    localparam int WEIGHT_WIDTH = 20;
    localparam int TRAIN_EPOCHS = 300;

    logic clk = 1'b0;
    logic rstn;

    logic [2:0] in_data;
    logic       in_data_vld;
    logic       in_data_rdy;
    logic       expected_result_data;
    logic       mode;
    logic [4*WEIGHT_WIDTH-1:0] init_weights_data;
    logic       init_weights_vld;
    logic       result_data;
    logic       result_vld;
    logic [4*WEIGHT_WIDTH-1:0] result_weights;

    int errors;
    integer fd;
    logic [2:0] tb_x;
    logic       tb_y;
    logic       tb_exp;

    always #5 clk = ~clk;

    neuron #(
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .LEARNING_RATE(20'sd512)
    ) dut (
        .clk(clk),
        .rstn(rstn),
        .in_data(in_data),
        .in_data_vld(in_data_vld),
        .in_data_rdy(in_data_rdy),
        .expected_result_data(expected_result_data),
        .mode(mode),
        .init_weights_data(init_weights_data),
        .init_weights_vld(init_weights_vld),
        .result_data(result_data),
        .result_vld(result_vld),
        .result_weights(result_weights)
    );

    function automatic logic and3_expected(input logic [2:0] x);
        and3_expected = &x;
    endfunction

    task automatic load_weights(
        input logic signed [WEIGHT_WIDTH-1:0] w0,
        input logic signed [WEIGHT_WIDTH-1:0] w1,
        input logic signed [WEIGHT_WIDTH-1:0] w2,
        input logic signed [WEIGHT_WIDTH-1:0] wbias
    );
        begin
            @(posedge clk);
            init_weights_data <= {wbias, w2, w1, w0};
            init_weights_vld  <= 1'b1;
            @(posedge clk);
            init_weights_vld  <= 1'b0;
            init_weights_data <= '0;
        end
    endtask

    task automatic send_sample(
        input  logic [2:0] x,
        input  logic       expected,
        input  logic       training_mode,
        output logic       y
    );
        begin
            @(posedge clk);
            while (!in_data_rdy) @(posedge clk);
            in_data              <= x;
            expected_result_data <= expected;
            mode                 <= training_mode;
            in_data_vld          <= 1'b1;
            @(posedge clk);
            in_data_vld          <= 1'b0;
            in_data              <= '0;
            expected_result_data <= 1'b0;
            wait (result_vld === 1'b1);
            y = result_data;
            @(posedge clk);
        end
    endtask

    task automatic print_weights;
        logic signed [WEIGHT_WIDTH-1:0] rw0, rw1, rw2, rb;
        begin
            rw0 = result_weights[0*WEIGHT_WIDTH +: WEIGHT_WIDTH];
            rw1 = result_weights[1*WEIGHT_WIDTH +: WEIGHT_WIDTH];
            rw2 = result_weights[2*WEIGHT_WIDTH +: WEIGHT_WIDTH];
            rb  = result_weights[3*WEIGHT_WIDTH +: WEIGHT_WIDTH];
            $display("Weights raw Q7.12: w0=%0d w1=%0d w2=%0d bias=%0d", rw0, rw1, rw2, rb);
            $display("weights.mem line: %05h%05h%05h%05h", rb[19:0], rw2[19:0], rw1[19:0], rw0[19:0]);
        end
    endtask

    initial begin
        errors = 0;
        rstn = 1'b0;
        in_data = '0;
        in_data_vld = 1'b0;
        expected_result_data = 1'b0;
        mode = 1'b0;
        init_weights_data = '0;
        init_weights_vld = 1'b0;

        repeat (5) @(posedge clk);
        rstn = 1'b1;
        repeat (2) @(posedge clk);

        load_weights(20'sd0, 20'sd0, 20'sd0, -20'sd10);

        $display("TRAINING START");
        for (int epoch = 0; epoch < TRAIN_EPOCHS; epoch++) begin
            for (int x = 0; x < 8; x++) begin
                tb_x = x[2:0];
                send_sample(tb_x, and3_expected(tb_x), 1'b1, tb_y);
            end
        end
        $display("TRAINING END");
        print_weights();

        $display("WORKING MODE CHECK");
        for (int x = 0; x < 8; x++) begin
            tb_x   = x[2:0];
            tb_exp = and3_expected(tb_x);
            send_sample(tb_x, tb_exp, 1'b0, tb_y);
            $display("x=%03b expected=%0d result=%0d", tb_x, tb_exp, tb_y);
            if (tb_y !== tb_exp) begin
                $error("Mismatch for x=%03b: expected=%0d result=%0d", tb_x, tb_exp, tb_y);
                errors++;
            end
        end

        fd = $fopen("weights.mem", "w");
        if (fd == 0) begin
            $error("Cannot open weights.mem for writing");
            errors++;
        end else begin
            $fwrite(fd, "%05h%05h%05h%05h\n",
                    result_weights[79:60], result_weights[59:40],
                    result_weights[39:20], result_weights[19:0]);
            $fclose(fd);
            $display("Saved trained weights to weights.mem");
        end

        if (errors == 0) begin
            $display("TEST PASSED");
        end else begin
            $fatal(1, "TEST FAILED: %0d errors", errors);
        end
        $finish;
    end
endmodule
