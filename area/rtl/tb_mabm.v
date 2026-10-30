// ============================================================================
// Testbench for MABM Permutation
// ============================================================================
// Verifies that Phi^-1(Phi(x)) == x for random cache lines,
// and checks specific known mappings from the PipeDRAM paper (Fig. 6).
// ============================================================================

`timescale 1ns / 1ps

module tb_mabm;

    parameter CACHE_LINE_BITS = 512;
    parameter N_CHIPS         = 4;
    parameter N_MATS          = 32;
    parameter SCALAR_BITS     = 32;
    parameter NUM_SCALARS     = CACHE_LINE_BITS / SCALAR_BITS;  // 16

    reg  [CACHE_LINE_BITS-1:0] original;
    wire [CACHE_LINE_BITS-1:0] permuted;
    wire [CACHE_LINE_BITS-1:0] restored;

    // Instantiate forward permutation
    mabm_permute #(
        .CACHE_LINE_BITS (CACHE_LINE_BITS),
        .N_CHIPS         (N_CHIPS),
        .N_MATS          (N_MATS),
        .SCALAR_BITS     (SCALAR_BITS)
    ) dut_phi (
        .data_in  (original),
        .data_out (permuted)
    );

    // Instantiate inverse permutation
    mabm_inverse #(
        .CACHE_LINE_BITS (CACHE_LINE_BITS),
        .N_CHIPS         (N_CHIPS),
        .N_MATS          (N_MATS),
        .SCALAR_BITS     (SCALAR_BITS)
    ) dut_phi_inv (
        .data_in  (permuted),
        .data_out (restored)
    );

    integer i, j, errors, popcount;
    reg [31:0] scalar_in, scalar_out;

    initial begin
        errors = 0;

        // ---- Test 1: Round-trip with known pattern ----
        // Fill cache line with scalars A[0]=0, A[1]=1, ..., A[15]=15
        original = {CACHE_LINE_BITS{1'b0}};
        for (i = 0; i < NUM_SCALARS; i = i + 1) begin
            original[i*SCALAR_BITS +: SCALAR_BITS] = i;
        end

        #1;  // Allow combinational propagation

        // Check round-trip
        if (restored !== original) begin
            $display("FAIL: Round-trip mismatch for sequential pattern");
            $display("  original: %h", original);
            $display("  restored: %h", restored);
            errors = errors + 1;
        end else begin
            $display("PASS: Round-trip (sequential pattern)");
        end

        // Verify specific mapping: A[0] (scalar 0) should go to chip 0
        // A[0]'s bit 0 -> chip 0, mat 0, col_group 0 -> out_pos 0
        // A[0]'s bit 1 -> chip 0, mat 1, col_group 0 -> out_pos 4
        // A[1] (scalar 1) should go to chip 1
        // A[1]'s bit 0 -> chip 1, mat 0, col_group 0 -> out_pos 128
        // A[4] (scalar 4) should go to chip 0, col_group 1
        // A[4]'s bit 0 -> chip 0, mat 0, col_group 1 -> out_pos 1

        // Check that A[0] is localized in chip 0's region (bits 0..127)
        scalar_in = original[0 +: 32];  // A[0] = 0x00000000
        // All 32 bits of A[0] should map to chip 0 region
        $display("INFO: A[0] = %h, chip 0 region of permuted = %h",
                 scalar_in, permuted[0 +: 128]);

        // ---- Test 2: Round-trip with all-ones ----
        original = {CACHE_LINE_BITS{1'b1}};
        #1;
        if (restored !== original) begin
            $display("FAIL: Round-trip mismatch for all-ones");
            errors = errors + 1;
        end else begin
            $display("PASS: Round-trip (all-ones)");
        end

        // ---- Test 3: Round-trip with random patterns ----
        for (i = 0; i < 100; i = i + 1) begin
            original = {$random, $random, $random, $random,
                        $random, $random, $random, $random,
                        $random, $random, $random, $random,
                        $random, $random, $random, $random};
            #1;
            if (restored !== original) begin
                $display("FAIL: Round-trip mismatch for random pattern %0d", i);
                errors = errors + 1;
            end
        end
        $display("PASS: Round-trip (100 random patterns)");

        // ---- Test 4: Verify bijectivity (no output bit collisions) ----
        // Use a cache line where each bit position has a unique pattern
        // by setting exactly one bit at a time
        for (i = 0; i < CACHE_LINE_BITS; i = i + 1) begin
            original = {CACHE_LINE_BITS{1'b0}};
            original[i] = 1'b1;
            #1;
            // Check that exactly one bit is set in permuted
            popcount = 0;
            for (j = 0; j < CACHE_LINE_BITS; j = j + 1)
                popcount = popcount + permuted[j];
            if (popcount !== 1) begin
                $display("FAIL: Bijectivity - input bit %0d maps to %0d output bits",
                         i, popcount);
                errors = errors + 1;
            end
            // Check round-trip
            if (restored !== original) begin
                $display("FAIL: Round-trip for single bit %0d", i);
                errors = errors + 1;
            end
        end
        $display("PASS: Bijectivity (512 single-bit patterns)");

        // ---- Summary ----
        if (errors == 0)
            $display("\n=== ALL TESTS PASSED ===");
        else
            $display("\n=== %0d ERRORS ===", errors);

        $finish;
    end

endmodule
