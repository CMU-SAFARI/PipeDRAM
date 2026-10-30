// ============================================================================
// MABM (Mat-Aware Bit Mapping) - Inverse Permutation (Phi^-1)
// ============================================================================
// Reconstructs the conventional cache line layout from MABM layout.
// Applied on the read path (DRAM -> host).
//
// For each position in the MABM layout indexed by (chip c, mat m, col_group l):
//   scalar index  i = l * N_CHIPS + c
//   bit index     j = m
//
// This is a fixed permutation: purely combinational wire routing.
// Expected synthesis result: 0 logic cells, 0 area.
// ============================================================================

module mabm_inverse #(
    parameter CACHE_LINE_BITS = 512,
    parameter N_CHIPS         = 4,
    parameter N_MATS          = 32,
    parameter SCALAR_BITS     = 32
)(
    input  wire [CACHE_LINE_BITS-1:0] data_in,   // MABM layout
    output wire [CACHE_LINE_BITS-1:0] data_out    // conventional layout
);

    localparam NUM_SCALARS   = CACHE_LINE_BITS / SCALAR_BITS;
    localparam BITS_PER_CHIP = CACHE_LINE_BITS / N_CHIPS;
    localparam COL_GROUPS    = NUM_SCALARS / N_CHIPS;

    // ---- Inverse permutation Phi^-1 ----
    genvar c, m, l;
    generate
        for (c = 0; c < N_CHIPS; c = c + 1) begin : gen_chip
            for (m = 0; m < N_MATS; m = m + 1) begin : gen_mat
                for (l = 0; l < COL_GROUPS; l = l + 1) begin : gen_col
                    // Input position in MABM layout
                    localparam integer IN_POS = c * BITS_PER_CHIP
                                              + m * COL_GROUPS
                                              + l;

                    // Reconstructed scalar and bit indices
                    localparam integer SCALAR_IDX = l * N_CHIPS + c;
                    localparam integer BIT_IDX    = m;

                    // Output position in conventional layout
                    localparam integer OUT_POS = SCALAR_IDX * SCALAR_BITS
                                              + BIT_IDX;

                    assign data_out[OUT_POS] = data_in[IN_POS];
                end
            end
        end
    endgenerate

endmodule
