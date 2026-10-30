# Dataset generation and provenance

## kmeans

The paper-scale Rodinia kmeans input contains 10,000 objects with 16,384
integer features per object. Each feature is in the range 0-255. The first
field on each line is an object identifier and is ignored by the workload.

Generate it from the included Rodinia `datagen.cpp` source with:

```bash
./scripts/generate_kmeans_input.sh
```

The script builds the generator and executes the equivalent of:

```bash
datagen 10000 16384 11
```

Seed 11 is fixed because it reproduces the convergence behavior of the paper
input: 11 clustering iterations with the default five clusters and threshold
0.001. The data values affect convergence, so an arbitrary random
seed does not necessarily reproduce the paper's operation count. The script
checks that exactly 10,000 objects were written and records a SHA-256 sidecar.
Full reproduction refuses an existing generated dataset when its sidecar does
not match.

The expected SHA-256 under Ubuntu 24.04 x86-64 with the included generator is:

```text
c4d6982efb1ea5354c5de841bd1192f1a6824f1cc21277b37e61221e83c9d899
```

The generated text file is approximately 558 MB and is not included in the
repository.

## heartwall

The paper-scale heartwall input is `test_4k.avi`, a 3840x2160 video. The
workflow processes 20 frames using eight threads:

```bash
./scripts/reproduce.sh full --heartwall-input /path/to/test_4k.avi
```

The accepted file must have this SHA-256:

```text
c51d82f46b5c95c4257d906ab1f399209ee511c73e87eea32340c7547a0f0d81
```

The workflow verifies the checksum before compiling or running any workload.
The video is not redistributed because its redistribution terms are unclear.
Without it, `full` mode keeps the paper's heartwall results, and validation
checks that they are unchanged.

## x264

x264 uses SPEC CPU2017 source and media, neither of which can be
redistributed. Its paper results are included, so all plotted applications and
geometric means use the paper's evaluation data.
