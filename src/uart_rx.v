module uart_rx
#(parameter CLKS_PER_BIT = 10417) // For 9600 baud with 100 MHz clock
(
    input        clk,         // System clock
    input        rx,          // UART RX line
    output reg [7:0] data,    // Received byte
    output reg   data_valid   // High when byte is ready
);

    localparam IDLE         = 3'b000;
    localparam START        = 3'b001;
    localparam DATA         = 3'b010;
    localparam STOP         = 3'b011;
    localparam CLEANUP      = 3'b100;

    reg [2:0]    state        = IDLE;
    reg [13:0]   clk_count    = 0;
    reg [2:0]    bit_index    = 0;
    reg [7:0]    rx_shift_reg = 0;

    always @(posedge clk) begin
        case (state)

            IDLE: begin
                data_valid <= 0;
                clk_count  <= 0;
                bit_index  <= 0;

                if (rx == 0) begin // Start bit detected
                    state <= START;
                end
            end

            START: begin
                if (clk_count == (CLKS_PER_BIT - 1)/2) begin
                    if (rx == 0) begin
                        clk_count <= 0;
                        state     <= DATA;
                    end else begin
                        state <= IDLE; // False start bit
                    end
                end else begin
                    clk_count <= clk_count + 1;
                end
            end

            DATA: begin
                if (clk_count < CLKS_PER_BIT - 1) begin
                    clk_count <= clk_count + 1;
                end else begin
                    clk_count <= 0;
                    rx_shift_reg[bit_index] <= rx;

                    if (bit_index < 7) begin
                        bit_index <= bit_index + 1;
                    end else begin
                        bit_index <= 0;
                        state     <= STOP;
                    end
                end
            end

            STOP: begin
                if (clk_count < CLKS_PER_BIT - 1) begin
                    clk_count <= clk_count + 1;
                end else begin
                    data       <= rx_shift_reg;
                    data_valid <= 1;
                    clk_count  <= 0;
                    state      <= CLEANUP;
                end
            end

            CLEANUP: begin
                state <= IDLE;
                data_valid <= 0;
            end

            default: state <= IDLE;
        endcase
    end
endmodule
