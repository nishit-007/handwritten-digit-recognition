`timescale 1ns / 1ps
module controller(
    input clk,             
    input rx,              
    output [6:0] seg,      
    output [3:0] an        
);

    wire [7:0] byte;
    wire       data_valid;

   
    uart_rx #(.CLKS_PER_BIT(10417)) uart_rx_inst (
        .clk(clk),
        .rx(rx),
        .data(byte),
        .data_valid(data_valid)
    );

    reg [3:0] digit;
    always @(posedge clk) begin
        if (data_valid)
            digit <= byte[3:0]; 
    end

    
    seven_seg_drive seg_drive (
        .nibble(digit),
        .segments(seg),
        .anodes(an)
    );
    assign an = 4'b1110; 
endmodule

