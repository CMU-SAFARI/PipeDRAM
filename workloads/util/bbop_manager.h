#ifndef BBOP_H
#define BBOP_H


#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <omp.h>
#include <stdint.h>


#ifndef NUM_THREADS
    #define NUM_THREADS 32
#endif

#ifndef OUT_APPEND
    #define OUT_APPEND 0
#endif

#ifndef INT4_DATA_TYPE
    #define INT4_DATA_TYPE 0
#endif

#define FLOAT_TO_INT(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))

#define DATATYPE_BBOP unsigned int

// Maximum number of distinct bbop_ids
#define MAX_BBOPS 100

#define SIMD_WIDTH 65536.0
#define SUBARRAYS 32
#define NUM_PARALLEL_BANKS 16   // BLP: banks working in parallel (= TRANSP_NUM_BANKS)

#define AAP_ENERGY 0.871
#define AAP_LATENCY 49 // 49 ns

// Transposition cost model parameters (same values as pud_simulator)
#define TRANSP_ROW_SIZE_BYTES     8192       // 8 KB row
#define TRANSP_SIMD_LANES         65536      // bits per row
#define TRANSP_CACHE_LINE_BYTES   64         // host cache line size
#define TRANSP_NUM_BANKS          16         // banks per chip (DDR4)
#define TRANSP_NUM_SUBARRAYS      32
#define TRANSP_ROWS_PER_SUBARRAY  512
#define TRANSP_CHIPS_PER_RANK     4
// Rank capacity: chips x banks x SAs x rows x row_size
#define TRANSP_RANK_CAPACITY_BYTES ((unsigned long long)TRANSP_CHIPS_PER_RANK \
    * TRANSP_NUM_BANKS * TRANSP_NUM_SUBARRAYS \
    * TRANSP_ROWS_PER_SUBARRAY * TRANSP_ROW_SIZE_BYTES)
// DDR4-2400 timing (ns)
#define TRANSP_T_RCD              15.0       // activate-to-column-access
#define TRANSP_T_RAS              32.0       // row active time (min ACT to PRE)
#define TRANSP_T_RP               15.0       // precharge latency
#define TRANSP_T_RC               (TRANSP_T_RAS + TRANSP_T_RP)  // 47 ns, row cycle (ACT to ACT same bank)
#define TRANSP_T_CWL              12.0       // CAS write latency
#define TRANSP_T_WR               15.0       // write recovery (before PRE)
#define TRANSP_T_CK               0.833      // clock period (DDR4-2400)
#define TRANSP_T_BURST            (8 * TRANSP_T_CK / 2)  // 3.33 ns, BL=8 data burst
// Per-workload bit precision
// Override at compile time with -DBBOP_BIT_PRECISION=8 (e.g., for x264 uint8 data).
// Default is 32-bit for PolyBench workloads (int/float -> fixed-point).
#ifndef BBOP_BIT_PRECISION
    #define BBOP_BIT_PRECISION    32
#endif
#define BBOP_BIT_PRECISION_HALF   (BBOP_BIT_PRECISION / 2)

#define TRANSP_BIT_PRECISION      BBOP_BIT_PRECISION
// DDR4-2400 energy model (Micron 4Gb x8, from ramulator-pim memspec)
// Per-device IDD currents: IDD0=60.75mA, IDD02=4.05mA, IDD2N=38.25mA,
//   IDD3N=44mA, IDD4R=184.5mA, IDD4W=168.75mA. VDD=1.2V, VPP=2.5V.
// Per-device per-command energy (DRAMPower formulas):
//   E_ACT = 982.3 pJ (VDD: 653.2 + VPP: 329.1)
//   E_PRE = 360.0 pJ
//   E_RD  = 562.0 pJ (column read burst)
//   E_WR  = 499.0 pJ (column write burst)
// I/O per device (DDR4 POD, R_drv=34 Ohm, RTT=48 Ohm, C_load=8pF):
//   724.2 pJ per burst (termination + switching + DQS)
// Per cache-line (8 x8 devices): multiply by 8.
#define TRANSP_E_ACT_PRE_NJ      10.74      // nJ per CL: 8 x (982.3+360.0) pJ
#define TRANSP_E_COL_RD_NJ        4.50      // nJ per CL: 8 x 562.0 pJ
#define TRANSP_E_COL_WR_NJ        3.99      // nJ per CL: 8 x 499.0 pJ
#define TRANSP_E_IO_NJ             5.79      // nJ per CL: 8 x 724.2 pJ
// Total per CL read:  21.03 nJ, per CL write: 20.52 nJ
// Round-trip (read+write): 41.55 nJ per CL = 81.2 pJ/bit
// L3 cache energy (CACTI, 22nm, 16 MB, 16-way)
#define TRANSP_E_L3_HIT_NJ        0.945      // nJ per CL: L3 hit (tag + data read)
#define TRANSP_E_L3_MISS_NJ       1.904      // nJ per CL: L3 miss (tag lookup + fill)
// Host-side cache line flush cost (CLFLUSH/CLWB + memory fence).
// Intel Xeon: ~80-100 ns per CLFLUSH.  Conservative estimate: 80 ns.
#define TRANSP_T_FLUSH_PER_CL     80.0       // ns per cache line flush (host-side)
#define TRANSP_LLC_SIZE_BYTES     (16ULL * 1024 * 1024)  // 16 MB LLC (typical server)

// PipeDRAM parameters
#define PIPEDRAM_SIMD_WIDTH       2048.0     // mat_columns(512) x num_chips(4)
#define PIPEDRAM_SUBARRAYS        32         // subarrays per bank (SALP)
#define PIPEDRAM_ENERGY_SCALE     (1.0 / 128.0)  // per-mat vs per-subarray

#define mmax(x, y) (x > y ? x : y)

// Create an enum to represent different operations
enum bbop_operation{
    BBOP_ADD,
    BBOP_ADD_8,
    BBOP_ADD_16,
    BBOP_ADD_32,
    BBOP_ADD_64,
    BBOP_SUB,
    BBOP_SUB_8,
    BBOP_SUB_16,
    BBOP_SUB_32,
    BBOP_SUB_64,
    BBOP_MUL,
    BBOP_MUL_8,
    BBOP_MUL_16,
    BBOP_MUL_32,
    BBOP_MUL_64,
    BBOP_DIV,
    BBOP_DIV_8,
    BBOP_DIV_16,
    BBOP_DIV_32,
    BBOP_DIV_64,
    BBOP_CPY,
    BBOP_RED,
    CPU_CPY,
    MUL_SSL_Sklansky_24,
    MUL_SSL_RB_39,
    MUL_SSL_RB_35,
    MUL_SSL_Sklansky_20,
    MUL_SSL_Sklansky_22,
    MUL_SSL_Sklansky_15,
    MUL_SSL_Sklansky_19,
    MUL_SSL_RB_34,
    MUL_SSL_RB_37,
    MUL_SSL_DIV_12,
    MUL_SSL_DIV_8,
    SUB_SSL_FA_11,
    MUL_SSL_Sklansky_18,
    SUB_SSL_FA_7,
    MUL_SSL_CS_14,
    SUB_SSL_FA_9,
    BBOP_RELU,
    BBOP_COUNT};
typedef enum bbop_operation bbop_operation;

typedef struct{
    int val;
    char pad[128];
} tvals;


struct bbop_statistic{
    bbop_operation operation;
    
    double simdram_1_subarray_latency;
    double simdram_64_subarray_latency;
    double simdram_64_subarray_dynamic_precision_latency;
    // Proteus (sequential + pipelined)
    double proteus_sequential_latency;
    double proteus_pipelined_latency;

    double simdram_1_subarray_energy;
    double simdram_64_subarray_energy;
    double simdram_64_subarray_dynamic_precision_energy;

    double proteus_sequential_energy;
    double proteus_pipelined_energy;

    // PipeDRAM
    double pipedram_sequential_latency;         // Base PipeDRAM (MABM only, no scheduling)
    double pipedram_pipelined_latency;          // + Modulo Scheduler (direct carry)
    double pipedram_hierarchical_latency;       // + Scheduler + Hierarchical Data Copy (full)

    double pipedram_sequential_energy;
    double pipedram_pipelined_energy;
    double pipedram_hierarchical_energy;

    // 16-bit precision variants
    double simdram_1_subarray_16_latency;
    double simdram_64_subarray_16_latency;
    double simdram_1_subarray_16_energy;
    double simdram_64_subarray_16_energy;

    double proteus_sequential_16_latency;
    double proteus_pipelined_16_latency;
    double proteus_sequential_16_energy;
    double proteus_pipelined_16_energy;

    double pipedram_sequential_16_latency;
    double pipedram_pipelined_16_latency;
    double pipedram_hierarchical_16_latency;
    double pipedram_sequential_16_energy;
    double pipedram_pipelined_16_energy;
    double pipedram_hierarchical_16_energy;

    long largest_input_a;
    long largest_input_b;
};
typedef struct bbop_statistic bbop_statistic;



// Transposition tracking (function-boundary model)
// Arrays crossing the kernel function boundary need transposition
// (horizontal<->vertical layout) for SIMDRAM.  PipeDRAM (MABM) avoids this.
#define MAX_TRANSPOSITION_ARRAYS 16384

typedef enum {
    BBOP_EXT_INPUT,     // External input:  needs H->V transposition at kernel entry
    BBOP_EXT_OUTPUT     // External output: needs V->H transposition at kernel exit
} bbop_array_role;

typedef struct {
    void                *ptr;           // array base pointer (for dedup)
    unsigned long long   num_elements;  // number of elements
    bbop_array_role      role;          // INPUT or OUTPUT
} bbop_transposition_entry;

// Create an array to store the bbop_statistic
void initialize_bbop_statistics();

// Mark that a bbop_id wrote to the given address range.
// Use after bbop_op_red() to track where the reduction result goes,
// enabling automatic MIMDRAM overlap detection across bbop_ids.
// Example: tmp[i][j] = bbop_op_red(..., 2);  bbop_mark_write(2, &tmp[i][j], 1);
void bbop_mark_write(int bbop_id, DATATYPE_BBOP *ptr, unsigned long long size);

// Register external arrays that cross the kernel function boundary.
// Call these at the start of the instrumented kernel, AFTER initialize_bbop_statistics().
void bbop_register_input (DATATYPE_BBOP *array, unsigned long long num_elements);
void bbop_register_output(DATATYPE_BBOP *array, unsigned long long num_elements);

// Create a bbop_op function
void bbop_op(bbop_operation operation, DATATYPE_BBOP *A, DATATYPE_BBOP *B, DATATYPE_BBOP *C, unsigned long long size, int bbop_id, bbop_operation daftpum_operation);

DATATYPE_BBOP bbop_op_red(bbop_operation operation, DATATYPE_BBOP *A, long size, int bbop_id);

void print_bbop_statistic();

unsigned short int float2fix(float n);


#endif
