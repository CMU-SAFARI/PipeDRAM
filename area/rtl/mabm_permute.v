// ============================================================================
// MABM (Mat-Aware Bit Mapping) - Forward Permutation (Phi)
// ============================================================================
// Implements the deterministic cache line reorganization described in
// PipeDRAM Section 4.2.  Applied on the write path (host -> DRAM).
//
// For each scalar A[i] with bit-position j in the input cache line:
//   target chip   c = i mod N_CHIPS
//   column group  l = floor(i / N_CHIPS)
//   target mat    m = j mod N_MATS
//
// The output cache line is organized so that the standard DDR burst
// places each bit into its target (chip, mat, column group).
//
// This is a fixed permutation: purely combinational wire routing.
// Expected synthesis result: 0 logic cells, 0 area.
// ============================================================================

module mabm_permute #(
    parameter CACHE_LINE_BITS = 512,   // 64 B cache line
    parameter N_CHIPS         = 4,     // DRAM chips per rank
    parameter N_MATS          = 32,    // mats per chip in PUD
    parameter SCALAR_BITS     = 32     // operand bit-width
)(
    input  wire [CACHE_LINE_BITS-1:0] data_in,   // conventional layout
    output wire [CACHE_LINE_BITS-1:0] data_out    // MABM layout
);

    // Derived constants
    localparam NUM_SCALARS   = CACHE_LINE_BITS / SCALAR_BITS;  // 16
    localparam BITS_PER_CHIP = CACHE_LINE_BITS / N_CHIPS;      // 128
    localparam COL_GROUPS    = NUM_SCALARS / N_CHIPS;           // 4

    // ---- Forward permutation Phi ----
    // Each assign is a single wire: data_out[q] = data_in[p]
    genvar s, b;
    generate
        for (s = 0; s < NUM_SCALARS; s = s + 1) begin : gen_scalar
            for (b = 0; b < SCALAR_BITS; b = b + 1) begin : gen_bit
                // Input position: scalar s, bit b
                localparam integer IN_POS  = s * SCALAR_BITS + b;

                // Target location in MABM layout
                localparam integer CHIP      = s % N_CHIPS;
                localparam integer COL_GROUP = s / N_CHIPS;
                localparam integer MAT       = b % N_MATS;

                // Output position: chip region + mat slot + column group
                localparam integer OUT_POS = CHIP * BITS_PER_CHIP
                                           + MAT  * COL_GROUPS
                                           + COL_GROUP;

                assign data_out[OUT_POS] = data_in[IN_POS];
            end
        end
    endgenerate

endmodule
