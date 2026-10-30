/* POLYBENCH/GPU-OPENMP
 *
 * This file is a part of the Polybench/GPU-OpenMP suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 * 
 * Copyright 2013, The University of Delaware
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "../util/timer.h"
#include "../util/bbop_manager.h"

/* Include polybench common header. */
#include "polybench.h"


/* Include benchmark-specific header. */
/* Default data type is double, default size is 512. */
#include "gramschmidt.h"

/* Array initialization. */
static
void init_array(int ni, int nj,
		DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj),
		DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      A[i][j] = ((DATA_TYPE) i*j) / ni;
      Q[i][j] = ((DATA_TYPE) i*(j+1)) / nj;
    }
  for (i = 0; i < nj; i++)
    for (j = 0; j < nj; j++)
      R[i][j] = ((DATA_TYPE) i*(j+2)) / nj;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj),
		 DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
	if (i % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
  for (i = 0; i < nj; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, R[i][j]);
	if (i % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, Q[i][j]);
	if (i % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gramschmidt(int ni, int nj,
			DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
			DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj),
			DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj))
{
  int i, j, k;

  #ifdef RUN_BASELINE
    printf("Running baseline\n");
    
    DATA_TYPE nrm;
    #pragma scop
    #pragma omp parallel for private (i, j)
    for (k = 0; k < _PB_NJ; k++){
      nrm = 0;
      for (i = 0; i < _PB_NI; i++)
        nrm += A[i][k] * A[i][k];
        R[k][k] = sqrt(nrm);
      
      for (i = 0; i < _PB_NI; i++)
	Q[i][k] = (R[k][k] > 0) ? A[i][k] / R[k][k] : 0; 
      
      
      for (j = k + 1; j < _PB_NJ; j++)
      {
        R[k][j] = 0;
        for (i = 0; i < _PB_NI; i++)
          R[k][j] += Q[i][k] * A[i][j];
        for (i = 0; i < _PB_NI; i++)
          A[i][j] = A[i][j] - Q[i][k] * R[k][j];
      }
    }
    #pragma endscop
  #endif

  #ifdef RUN_PIM
    printf("Running PIM\n");

    // Register external arrays for transposition tracking.
    // A is a kernel parameter (external input).
    // Q and R are kernel parameters (external outputs).
    bbop_register_input (A[0], (unsigned long long)_PB_NI * _PB_NJ);  // A[NI][NJ]
    bbop_register_output(Q[0], (unsigned long long)_PB_NI * _PB_NJ);  // Q[NI][NJ]
    bbop_register_output(R[0], (unsigned long long)_PB_NJ * _PB_NJ);  // R[NJ][NJ]

    DATA_TYPE nrm;
    DATA_TYPE tmp1[_PB_NI];
    DATA_TYPE transposed_a[_PB_NI];
    DATA_TYPE array_for_r[_PB_NI];
    DATA_TYPE transposed_q[_PB_NI];
    

    for (k = 0; k < _PB_NJ; k++){
      #ifndef BBOP_SKIP_COMPUTE
      for (i = 0; i < _PB_NI; i++){
        transposed_a[i] = A[i][k];
      }
      #endif

      bbop_op(BBOP_MUL, transposed_a, transposed_a, tmp1, _PB_NI, 0, MUL_SSL_Sklansky_22);
      nrm = bbop_op_red(BBOP_ADD, tmp1, _PB_NI, 1);

      #ifndef BBOP_SKIP_COMPUTE
      for (i = 0; i < _PB_NI; i++){
        array_for_r[i] = R[k][k];
      }
      #endif

      bbop_op(BBOP_DIV, transposed_a, array_for_r, transposed_q, _PB_NI, 2, MUL_SSL_DIV_12);

      #ifndef BBOP_SKIP_COMPUTE
      for (i = 0; i < _PB_NI; i++){
        Q[i][k] = transposed_q[i];
      }
      #endif

      DATA_TYPE transposed_a_j[_PB_NI];
      DATA_TYPE output_qa[_PB_NI];
      DATA_TYPE rkj_array[_PB_NI];
      DATA_TYPE output_qr[_PB_NI];

      for (j = k + 1; j < _PB_NJ; j++)
      {
        #ifndef BBOP_SKIP_COMPUTE
        // Extract column j of A
        for (i = 0; i < _PB_NI; i++)
          transposed_a_j[i] = A[i][j];
        #endif

        // R[k][j] = sum_i Q[i][k] * A[i][j]  (column dot product)
        bbop_op(BBOP_MUL, transposed_q, transposed_a_j, output_qa, _PB_NI, 3, MUL_SSL_Sklansky_22);
        R[k][j] = bbop_op_red(BBOP_ADD, output_qa, _PB_NI, 4);

        #ifndef BBOP_SKIP_COMPUTE
        // Broadcast R[k][j] into an array for elementwise ops
        for (i = 0; i < _PB_NI; i++)
          rkj_array[i] = R[k][j];
        #endif

        // A[:,j] = A[:,j] - Q[:,k] * R[k][j]  (rank-1 update)
        bbop_op(BBOP_MUL, transposed_q, rkj_array, output_qr, _PB_NI, 5, MUL_SSL_Sklansky_22);
        bbop_op(BBOP_SUB, transposed_a_j, output_qr, transposed_a_j, _PB_NI, 6, SUB_SSL_FA_11);

        #ifndef BBOP_SKIP_COMPUTE
        // Write back column j of A
        for (i = 0; i < _PB_NI; i++)
          A[i][j] = transposed_a_j[i];
        #endif
      }

    }
  #endif 
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  
  #ifdef RUN_BASELINE
    omp_set_num_threads(THREADS);
  #endif 

  #ifdef RUN_PIM
    omp_set_num_threads(NUM_THREADS);
    initialize_bbop_statistics();
  #endif

  int n_threads = omp_get_max_threads();
  printf("Number of threads: %d\n", n_threads);

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,NJ,NJ,nj,nj);
  POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,NI,NJ,ni,nj);

  /* Initialize array(s). */
  init_array (ni, nj,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(R),
	      POLYBENCH_ARRAY(Q));

  #ifdef GET_TIME
    TimerEnergy te;
    startTimerEnergy(&te);
  #endif

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gramschmidt (ni, nj,
          POLYBENCH_ARRAY(A),
          POLYBENCH_ARRAY(R),
          POLYBENCH_ARRAY(Q));

  #ifdef GET_TIME
    stopTimerEnergy(&te);
    #ifdef RUN_BASELINE
      printTimerEnergy(te);
    #endif
  #endif

  #ifdef RUN_PIM
    print_bbop_statistic();
  #endif
  
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

  return 0;
}
