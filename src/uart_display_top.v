`timescale 1ns / 1ps
module controller(
    input clk,             // 100 MHz clock (Basys3)
    input rx,              // UART RX
    output [6:0] seg,      // Seven segment output (a-g)
    output [3:0] an        // Anodes (only one active)
);

    wire [7:0] byte;
    wire       data_valid;

    // UART Receiver with 9600 baud (100 MHz / 9600 = ~10417)
    uart_rx #(.CLKS_PER_BIT(10417)) uart_rx_inst (
        .clk(clk),
        .rx(rx),
        .data(byte),
        .data_valid(data_valid)
    );

    reg [3:0] digit;
    always @(posedge clk) begin
        if (data_valid)
            digit <= byte[3:0];  // take only lower nibble
    end

    // Seven segment driver
    seven_seg_drive seg_drive (
        .nibble(digit),
        .segments(seg),
        .anodes(an)
    );
    assign an = 4'b1110; // Enable only one digit (rightmost)
endmodule
