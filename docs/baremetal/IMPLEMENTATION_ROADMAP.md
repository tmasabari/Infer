# BareMetalInfer — Implementation Roadmap

**Approach:** prove a bootable, testable CPU inference vertical slice before expanding model support or platform count.

## Phase 0 — Architecture and feasibility

- Freeze CPU-only and no-host-OS runtime constraints.
- Confirm x86-64 boot protocol and QEMU baseline.
- Validate Rust `no_std` target, linker script and panic path.
- Define HAL, allocator, CPU feature and worker contracts.
- Investigate AArch64 and Apple Silicon boot constraints separately.

**Exit:** reproducible image boots in QEMU, reports memory map and CPU features, and reaches a minimal console.

## Phase 1 — Bare-metal runtime foundation

- Boot entry, GDT/IDT, exception handling and page tables.
- Physical memory map parser and bounded page allocator.
- Heap/arena allocator with accounting and guard checks.
- Timer and serial/framebuffer console.
- CPU topology discovery and secondary-core startup.
- Basic worker pool, barriers and atomics.

**Exit:** memory and worker stress tests pass; faults are diagnosable; no hosted OS APIs or libc symbols are linked.

## Phase 2 — Inference core without model-specific optimization

- `no_std` domain types and error model.
- Tensor descriptors, dtype and quantization metadata.
- Scalar math, normalization, RoPE and activation reference operations.
- Matrix-vector and matrix-matrix reference kernels.
- Tokenizer interface and model source/module registry.
- Contiguous KV-cache and sequence lifecycle.

**Exit:** host-side unit tests and QEMU target tests validate core components independently of boot hardware.

## Phase 3 — First end-to-end small model

- Select a documented small dense decoder model and checkpoint format.
- Build an offline model packer/exporter.
- Load model weights from a boot module into validated read-only regions.
- Implement embedding, attention, MLP, residual, normalization and output head.
- Implement greedy decoding and deterministic sampling.
- Add interactive serial console first; VGA/keyboard are optional presentation layers.

**Exit:** complete prompt-to-token generation on one CPU core with reference-logit and token comparisons.

## Phase 4 — SIMD and multicore optimization

- x86-64 baseline and AVX2/FMA kernels.
- AArch64 NEON kernels.
- Optional AVX-512/SVE only after baseline profiling.
- Fused projections and quantization-aware GEMV/GEMM.
- Multicore layer/projection scheduling and reduction correctness.
- Separate prefill and decode kernel plans.
- Huge-page experiments with controlled A/B tests.

**Exit:** reproducible improvement over scalar baseline without correctness regressions; scaling and saturation documented.

## Phase 5 — Model and format expansion

- Qwen-family architecture adapter after exact architecture validation.
- Q8_0 and additional quantization formats selected by measured CPU benefit.
- Robust tokenizer assets and Unicode tests.
- Block/paged KV only if memory fragmentation or context scaling requires it.
- Model compatibility matrix and golden test corpus.

**Exit:** at least one additional model/checkpoint runs without changing platform HAL or kernel dispatch contracts.

## Phase 6 — Platform expansion

- Intel hardware validation.
- AArch64 QEMU and physical hardware.
- Apple Silicon feasibility decision and platform-specific boot research.
- Additional firmware/boot protocol adapters without changing inference core.

## Required engineering gates

Every phase includes:
- Reproducible build and image generation.
- No accidental `std`, libc, OS syscall or vendor SDK dependency in the target.
- Static binary inspection and symbol/dependency audit.
- Unit tests for scalar reference operations.
- Contract tests for HAL and allocator implementations.
- Fault-path and memory exhaustion tests.
- Numerical differential tests for optimized kernels.
- Benchmark metadata and repeatable prefill/decode workloads.

## First milestone definition of done

A bootable x86-64 image starts in QEMU, initializes memory and one CPU core, loads a tiny dense decoder model supplied as a boot module, accepts a prompt over serial, generates tokens using scalar Rust kernels, and exits no operation through an OS service. The image has no dynamic libraries, libc, GPU runtime, GPU driver, vendor compute SDK or host-OS runtime dependency.
