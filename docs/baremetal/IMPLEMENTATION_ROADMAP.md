# BareMetalInfer — Implementation Roadmap

**Confirmed scope:** x86-64 only; AMD and Intel; small dense model first; inference only; simple bounded memory; two boot paths, implemented sequentially.

## Phase 0 — Build and boot skeleton

- Create minimal Rust `no_std` x86-64 kernel target and linker script.
- Establish QEMU + standard-loader boot path.
- Define shared `BootInfo` contract and validate memory-map/module ranges.
- Add serial output and panic/exception reporting.
- Add static binary and forbidden-dependency audit.

**Exit:** reproducible QEMU image boots through the standard loader, reports validated memory regions and CPU identity/features, then halts cleanly. No libc, dynamic linker, OS syscall or vendor runtime dependency.

## Phase 1 — Minimal runtime and bounded memory

- Parse and normalize boot memory map.
- Reserve kernel, stacks, model module, KV, scratch and metadata regions.
- Implement aligned bump/arena allocator with bounds and high-water accounting.
- Add only the page tables and physical page allocation needed for x86-64 operation.
- Initialize GDT/IDT, exception handlers and monotonic timer.
- Add one-core runtime first; defer AP startup until the single-core inference vertical slice works.

**Exit:** allocator tests cover alignment, exhaustion, overflow and reserved-range protection. Faults report through serial. No general-purpose VM, swap or paged allocator.

## Phase 2 — Scalar inference foundation

- Define tensor shape/stride/dtype and checked indexing.
- Define quantization metadata and packed-model tensor descriptors.
- Implement scalar reference matvec/matmul, normalization, RoPE, activation and softmax.
- Define model source, tokenizer, sampler and KV-cache interfaces.
- Implement bounded contiguous KV-cache and one active sequence.
- Build host-side numerical tests for reference operators.

**Exit:** reference operators pass deterministic unit tests and target-side smoke tests. Memory use is bounded and reported.

## Phase 3 — First end-to-end small model

- Select one small dense decoder checkpoint with known architecture and permissive redistribution terms.
- Implement a host-side exporter/packer that emits versioned, aligned model sections.
- Validate header, tensor table, offsets, dimensions, quantization and checksums before inference.
- Load packed weights from a boot module as read-only data without unnecessary copies.
- Implement embedding, decoder blocks, attention, MLP, final norm and output projection.
- Integrate the exact tokenizer vocabulary/merges and Unicode test corpus.
- Implement greedy decode first; add top-p only after deterministic baseline.
- Use serial input/output as the initial user interface.

**Exit:** prompt-to-token generation works on one x86-64 core in QEMU and at least one physical AMD or Intel machine. Logits and greedy tokens match the reference within documented tolerances.

## Phase 4 — x86 SIMD and performance

- Add conservative x86-64 SIMD baseline.
- Add AVX2/FMA quantized matvec and matmul kernels.
- Add optional AVX-512 kernels behind runtime feature checks.
- Fuse compatible projections only after baseline kernels are correct.
- Separate prefill and decode dispatch.
- Benchmark scalar vs SIMD and collect memory bandwidth/latency counters where available.

**Exit:** each optimized kernel passes differential tests and has a repeatable performance improvement on at least one supported CPU without regressions on fallback CPUs.

## Phase 5 — Multicore and second boot path

- Discover ACPI processor topology and local APIC capabilities.
- Start secondary cores using a small x86 AP startup path.
- Add bounded worker pool, barriers and deterministic reduction behavior.
- Measure scaling and saturation; avoid oversubscription.
- Add second boot path (direct UEFI/firmware entry) and normalize into the same `BootInfo`.
- Run identical inference tests through both boot paths.

**Exit:** both boot paths execute the same model and inference core; multicore output correctness is stable across worker counts.

## Phase 6 — Model and quantization expansion

- Add one additional small dense model only after adapter boundaries prove useful.
- Add Q8_0 and another quantization format selected by measured CPU performance.
- Expand context and KV sizing tests.
- Add model compatibility documentation and golden prompts.
- Consider more advanced memory/KV strategies only when measurements demonstrate need.

**Exit:** a second model runs without changing boot/runtime services or duplicating kernels.

## Work-package acceptance gates

All packages must include:
- Explicit interfaces and ownership.
- File-level changes.
- Unit/contract tests.
- QEMU integration test where applicable.
- Static binary/dependency audit.
- Failure-path tests (invalid boot info, memory exhaustion, unsupported ISA).
- Numerical comparison for math/kernel changes.
- Benchmark baseline for performance changes.

## First end-to-end milestone

A QEMU x86-64 image boots using the standard loader, initializes a single CPU core and serial console, validates a boot-provided packed tiny dense decoder model, accepts a prompt over serial and generates tokens with scalar Rust kernels and a contiguous KV-cache. No operation relies on an OS service, libc, GPU, driver or vendor compute SDK. This milestone deliberately precedes the custom/direct boot path and multicore support.

## Phase -1 — Isolated development environment

Before implementation, provision the documented Ubuntu 26.04 host environment using KVM/QEMU/libvirt and an Ubuntu Server development VM. This does not change the host OS or boot configuration.

- Create the `BareMetalInfer-Dev` VM using Q35, OVMF, host-passthrough CPU, 6 vCPUs initially, 24–32 GiB RAM and a 100 GiB QCOW2 disk.
- Verify KVM, guest CPUID feature visibility and actual vCPU topology.
- Install the pinned Rust, LLVM/Clang, NASM, Ninja, QEMU and test toolchain.
- Establish host-launched QEMU/KVM for freestanding image tests; do not require nested virtualization.
- Keep hosted Linux tests and freestanding guest tests as separate benchmark categories.

**Exit:** the development VM is reproducible and accessible; host QEMU/KVM can launch a minimal freestanding smoke image; the host OS and boot configuration remain unchanged.

## Phase 0A — Hosted reference backend (development aid)

Build a minimal Linux-hosted adapter for the shared inference core so model loading, tokenizer, scalar kernels, SIMD kernels and benchmark tooling can be developed quickly in the VM. This backend is a test/development target only and must not introduce `std`, libc, syscalls or OS dependencies into the freestanding artifact.

**Exit:** the same core numerical and model tests can run in hosted test mode and in freestanding/QEMU mode. Benchmark output identifies the execution environment explicitly.


## Architecture decision update — 2026-11-27

The detailed source comparison and rationale are maintained in [ARCHITECTURE_COMPARISON_AND_DECISION.md](ARCHITECTURE_COMPARISON_AND_DECISION.md). This section clarifies the roadmap above where its generic "small dense decoder" wording is insufficient.

### Primary model target

The intended performance MVP is **Qwen3.5-0.8B with H128/Q4-G32-DOT4 quantization**, using the exact validated artifact and tokenizer. The currently evaluated artifact is approximately 425 MB. Qwen3-0.6B Q8_0 from `llm.asm` may be used as an optional boot/platform smoke fixture, but it is not the primary model target.

### Updated implementation sequence

1. **Environment and source audit:** establish hosted Qwen3.5 reference execution, pin source revisions and licenses, hash the model artifact, and capture a matched baseline.
2. **Freestanding platform skeleton:** keep the existing Rust `no_std` plan; use an existing standard bootloader and adopt relevant low-level design patterns from `llm.asm`. Do not build an OS or custom bootloader.
3. **Model contract and scalar reference:** add explicit H128/Q4-G32-DOT4 descriptors, checked tensor/model metadata and the Qwen3.5 block topology. Implement scalar correctness for embeddings, normalization, projections, MLP, Gated DeltaNet, convolution history and full attention.
4. **Qwen3.5 state and generation:** keep recurrent/convolution state distinct from attention KV; implement tokenizer, prefill, decode and greedy sampling for one sequence.
5. **SIMD optimization:** add AVX2 and optional AVX-512/VNNI paths only after scalar differential tests. Benchmark prefill and decode independently.
6. **Freestanding end-to-end MVP:** boot, load the exact model, prefill, decode and compare logits/tokens with the hosted oracle.
7. **SMP and further boot path:** add APIC/SMP after single-core correctness; add direct firmware boot after the standard-loader inference image is stable.

### Revised phase interpretation

- Existing generic dense-decoder WP-09 is a useful architecture bring-up/reference exercise. It does **not** by itself satisfy the Qwen3.5 MVP.
- The first model release requires Qwen3.5 hybrid-state support and H128/Q4-G32-DOT4; do not silently downgrade the model to Qwen3-0.6B or Q8_0 to claim completion.
- The 150+ generated tokens/sec objective is a stretch target. The current local native results are a baseline to reproduce, not a guaranteed bare-metal outcome.
- Keep the original memory and platform constraints: one sequence, contiguous bounded allocations, no paging/paged KV, no OS services and no direct boot path in the first end-to-end milestone.

