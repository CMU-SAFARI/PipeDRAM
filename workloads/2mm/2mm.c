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
#include <stdint.h>

#include "../util/timer.h"
#include "../util/bbop_manager.h"

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "2mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nl),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  *alpha = 32412;
  *beta = 2123;
  for (i = 0; i < ni; i++){
    for (j = 0; j < nk; j++){
      DATA_TYPE tmp = ((DATA_TYPE) i*j) / ni; 
      // if INT4_DATA_TYPE is 1, make sure that tmp is between 0 and 15
      if (INT4_DATA_TYPE == 1){
        tmp = tmp % 16;
      }
      
      A[i][j] = tmp;
    }
  }

  for (i = 0; i < nk; i++){
    for (j = 0; j < nj; j++){
      DATA_TYPE tmp = ((DATA_TYPE) i*(j+1)) / nj;

      // if INT4_DATA_TYPE is 1, make sure that tmp is between 0 and 15
      if (INT4_DATA_TYPE == 1){
        tmp = tmp % 16;
      }

      B[i][j] = tmp;
    }
  }

  for (i = 0; i < nl; i++){
    for (j = 0; j < nj; j++){
      DATA_TYPE tmp = ((DATA_TYPE) i*(j+3)) / nl;

      // if INT4_DATA_TYPE is 1, make sure that tmp is between 0 and 15
      if (INT4_DATA_TYPE == 1){
        tmp = tmp % 16;
      }
      C[i][j] = tmp;
    }
  }

  for (i = 0; i < ni; i++){
    for (j = 0; j < nl; j++){
      DATA_TYPE tmp = ((DATA_TYPE) i*(j+2)) / nk;

      // if INT4_DATA_TYPE is 1, make sure that tmp is between 0 and 15
      if (INT4_DATA_TYPE == 1){
        tmp = tmp % 16;
      }

      D[i][j] = tmp;
    }
  }

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, D[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_2mm(int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j, k;

// check if I should run the BASELINE or PIM version
#ifdef RUN_BASELINE 
    printf("Running baseline version\n");

    #pragma scop
    /* D := alpha*A*B*C + beta*D */
    #pragma omp parallel
    {
      #pragma omp for private (j, k)
      for (i = 0; i < _PB_NI; i++)
        for (j = 0; j < _PB_NJ; j++)
      {
          tmp[i][j] = 0;
        for (k = 0; k < _PB_NK; ++k)
        tmp[i][j] += alpha * A[i][k] * B[k][j];
          }
      #pragma omp for private (j, k)
      for (i = 0; i < _PB_NI; i++)
        for (j = 0; j < _PB_NL; j++)
          {
      D[i][j] *= beta;
      for (k = 0; k < _PB_NJ; ++k)
        D[i][j] += tmp[i][k] * C[k][j];
    }
    }
    #pragma endscop
# endif

#ifdef RUN_PIM
    printf("[DEBUG] Running PIM version\n");

    // Register external arrays for transposition tracking.
    // Function boundary: A, B, C, D are kernel parameters (external).
    // tmp is allocated outside but only used as intermediate within the kernel.
    bbop_register_input (A[0], _PB_NI * _PB_NK);  // A[NI][NK]
    bbop_register_input (B[0], _PB_NK * _PB_NJ);  // B[NK][NJ]
    bbop_register_input (C[0], _PB_NL * _PB_NJ);  // C[NL][NJ]
    bbop_register_input (D[0], _PB_NI * _PB_NL);  // D[NI][NL] (read before modification)
    bbop_register_output(D[0], _PB_NI * _PB_NL);  // D[NI][NL] (result written back)
    bbop_register_output(tmp[0], _PB_NI * _PB_NJ);  // tmp[NI][NJ] (PUD writes in loop 1)
    bbop_register_input (tmp[0], _PB_NI * _PB_NJ);  // tmp[NI][NJ] (PUD reads back in loop 2)

    DATA_TYPE alpha_array[_PB_NK];
    DATA_TYPE beta_array[_PB_NL];

    // alpha needs to become an array
    for (k = 0; k < _PB_NK; ++k){
      alpha_array[k] = alpha;
    }

    // beta needs to become an array
    for (j = 0; j < _PB_NL; j++){
      beta_array[j] = beta;
    }

    // Create a temporary array to store the result of A*B
    DATA_TYPE output_a_b[_PB_NK];
    DATA_TYPE output_a_b_alpha[_PB_NK];
    //Create a temporary array to store the result of tmp*C
    DATA_TYPE output_tmp_c[_PB_NJ];   
    
    // B needs to be transposed
    DATA_TYPE transposed_b[_PB_NK];
    DATA_TYPE transposed_c[_PB_NJ];

    /* D := alpha*A*B*C + beta*D */
    // get the time before the parallel region
    double start = omp_get_wtime();
    for (i = 0; i < _PB_NI; i++){
      for (j = 0; j < _PB_NJ; j++){           
            #ifndef BBOP_SKIP_COMPUTE
            for (k = 0; k < _PB_NK; ++k){
                transposed_b[k] = B[k][j];
            }
            #endif

            bbop_op(BBOP_MUL, A[i], transposed_b, output_a_b, _PB_NK, 0, MUL_SSL_Sklansky_24);
            bbop_op(BBOP_MUL, output_a_b, alpha_array, output_a_b_alpha, _PB_NK, 1, MUL_SSL_RB_39);
            tmp[i][j] =  bbop_op_red(BBOP_ADD, output_a_b_alpha, _PB_NK, 2);
            
        }
    }

    for (i = 0; i < _PB_NI; i++){
      bbop_op(BBOP_MUL, D[i], beta_array, D[i], _PB_NL, 3, MUL_SSL_RB_35);
      
      for (j = 0; j < _PB_NL; j++){
        #ifndef BBOP_SKIP_COMPUTE
        // C needs to be transposed
        for (k = 0; k < _PB_NJ; ++k){
          transposed_c[k] = C[k][j];
        }
        #endif

        bbop_op(BBOP_MUL, tmp[i], transposed_c, output_tmp_c, _PB_NJ, 4, MUL_SSL_Sklansky_20);
        D[i][j] = bbop_op_red(BBOP_ADD, output_tmp_c, _PB_NJ, 5);
      }
    }

    // get the time after the parallel region
    double end = omp_get_wtime();
    printf("[DEBUG] Total time: %f\n", end - start);

#endif

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  #ifdef RUN_BASELINE
    omp_set_num_threads(THREADS);
  #endif 

  #ifdef RUN_PIM
    omp_set_num_threads(NUM_THREADS);
    initialize_bbop_statistics();
  #endif

  int n_threads = omp_get_max_threads();
  printf("Number of threads: %d\n", n_threads);

  if (INT4_DATA_TYPE == 1){
    printf("Running with INT4_DATA_TYPE\n");
  }
  
  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;

  POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NL,NJ,nl,nj);
  POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));
  
  #ifdef GET_TIME
    TimerEnergy te;
    startTimerEnergy(&te);
  #endif

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_2mm (ni, nj, nk, nl,
        alpha, beta,
        POLYBENCH_ARRAY(tmp),
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B),
        POLYBENCH_ARRAY(C),
        POLYBENCH_ARRAY(D));

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
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(D)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
	
  return 0;
}
