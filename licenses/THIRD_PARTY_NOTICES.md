# Third-party notices

The MIT license at the package root applies only to project-authored simulator,
plotting, workflow, and validation code.

The benchmark sources under `workloads/` retain their original headers and
notices:

- PolyBench/C kernels and utilities: copyright notices and redistribution
  terms are embedded in the copied sources and per-workload README files.
- Rodinia backprop, heartwall, and kmeans: original suite notices remain in
  their source files. The heartwall AVI support code is GPL-licensed; its
  complete notice is retained in `workloads/heartwall/AVI/avilib.h`.
- kmeans compatibility code retains the GNU C Library license notice.
- Phoenix PCA retains the Stanford BSD license text in the source headers and
  workload README.

No SPEC CPU2017 source or media is included. Only numerical results produced
during the paper evaluation are redistributed.
