CC = gcc
CFLAGS = -fopenmp -lm -O3 -w

.PHONY: all

all: gramschmidt_pim.exe gramschmidt_pim_fast.exe gramschmidt_baseline.exe

gramschmidt_pim.exe: gramschmidt.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DPAPER_DATASET -DRUN_PIM -o $@ $(CFLAGS)

gramschmidt_pim_fast.exe: gramschmidt.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DPAPER_DATASET -DRUN_PIM -DBBOP_SKIP_COMPUTE -o $@ $(CFLAGS)

gramschmidt_baseline.exe: gramschmidt.c polybench.c ../util/bbop_manager.c
	$(CC) $^ -DGET_TIME -DPAPER_DATASET -DRUN_BASELINE -o $@ $(CFLAGS)

clean:
	rm -f gramschmidt_pim.exe gramschmidt_pim_fast.exe gramschmidt_baseline.exe
