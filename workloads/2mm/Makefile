CC = gcc
CFLAGS = -fopenmp -lm -Ofast -march=native -funroll-loops

.PHONY: all clean

all: 2mm_pim.exe 2mm_baseline.exe 2mm_pim_int4.exe

2mm_pim.exe: 2mm.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DPAPER_DATASET -DRUN_PIM -o $@ $(CFLAGS)

2mm_pim_fast.exe: 2mm.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DPAPER_DATASET -DRUN_PIM -DBBOP_SKIP_COMPUTE -o $@ $(CFLAGS)

2mm_pim_int4.exe: 2mm.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DPAPER_DATASET -DINT4_DATA_TYPE -DRUN_PIM -o $@ $(CFLAGS)

2mm_baseline.exe: 2mm.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DGET_TIME -DPAPER_DATASET -DRUN_BASELINE -DTHREADS=32 -o $@ $(CFLAGS)

clean:
	rm -f 2mm_pim.exe 2mm_pim_fast.exe 2mm_baseline.exe 2mm_pim_int4.exe
