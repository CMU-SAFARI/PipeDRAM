CC = gcc
CFLAGS = -lpthread -lm -O3 -w  -fopenmp

.PHONY: all

all: pca_pim.exe pca_pim_fast.exe pca_baseline.exe

pca_pim.exe: pca-pthread.c ../util/bbop_manager.c
	$(CC) $^ -DRUN_PIM -o $@ $(CFLAGS)

pca_pim_fast.exe: pca-pthread.c ../util/bbop_manager.c
	$(CC) $^ -DRUN_PIM -DBBOP_SKIP_COMPUTE -o $@ $(CFLAGS)

pca_baseline.exe: pca-pthread.c ../util/bbop_manager.c
	$(CC) $^ -DGET_TIME -DRUN_BASELINE -o $@ $(CFLAGS)

clean:
	rm -f pca_pim.exe pca_pim_fast.exe pca_baseline.exe
