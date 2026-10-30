// ============================================================================
// MABM Top-Level - Write path (Phi) + Read path (Phi^-1)
// ============================================================================
// Wraps both permutations into a single module for synthesis.
// Both paths are pure combinational wire routing.
// ============================================================================

module mabm_top #(
    parameter CACHE_LINE_BITS = 512,
    parameter N_CHIPS         = 4,
    parameter N_MATS          = 32,
    parameter SCALAR_BITS     = 32
)(
    // Write path: host -> DRAM (apply Phi)
    input  wire [CACHE_LINE_BITS-1:0] write_data_in,
    output wire [CACHE_LINE_BITS-1:0] write_data_out,

    // Read path: DRAM -> host (apply Phi^-1)
    input  wire [CACHE_LINE_BITS-1:0] read_data_in,
    output wire [CACHE_LINE_BITS-1:0] read_data_out
);

    mabm_permute #(
        .CACHE_LINE_BITS (CACHE_LINE_BITS),
        .N_CHIPS         (N_CHIPS),
        .N_MATS          (N_MATS),
        .SCALAR_BITS     (SCALAR_BITS)
    ) u_phi (
        .data_in  (write_data_in),
        .data_out (write_data_out)
    );

    mabm_inverse #(
        .CACHE_LINE_BITS (CACHE_LINE_BITS),
        .N_CHIPS         (N_CHIPS),
        .N_MATS          (N_MATS),
        .SCALAR_BITS     (SCALAR_BITS)
    ) u_phi_inv (
        .data_in  (read_data_in),
        .data_out (read_data_out)
    );

endmodule
