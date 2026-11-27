# BareMetalInfer — Bare-Metal Architecture and Inference Engine Comparison

**Status:** Architecture decision record / research baseline  
**Branch:** `BareMetalInfer`  
**Date:** 2026-11-27  
**Decision:** Use a Rust `no_std` inference runtime with a boot/platform design informed by `alesha-pro/llm.asm` and a Qwen3.5 inference design informed by `Danmoreng/qwen35-cpu`. Do not adopt either repository wholesale.

## 1. Executive decision

The best direction is a **selective hybrid**, not a source-code merge:

1. **Bare-metal platform and boot:** use the small, direct x86-64 boot/runtime architecture demonstrated by [alesha-pro/llm.asm](https://github.com/alesha-pro/llm.asm) as the primary reference for boot handoff, long-mode setup, physical memory layout, CPU initialization, serial diagnostics, ACPI/APIC discovery and multicore startup. Prefer an existing standard bootloader; do not create a general-purpose OS or custom bootloader as an MVP prerequisite.
2. **Model and inference semantics:** use [Danmoreng/qwen35-cpu](https://github.com/Danmoreng/qwen35-cpu) as the primary reference for Qwen3.5-0.8B, its hybrid attention/DeltaNet state, H128/Q4-G32-DOT4 quantization, CPU kernel behavior, KV/state lifecycle and numerical expectations.
3. **Target implementation:** retain the branch's Rust `no_std` target and narrow x86-64 assembly boundary. Keep a hosted Linux reference/test adapter for fast development. Port only the required model semantics and measured hot kernels into the freestanding runtime; do not pull C++ standard-library or OS dependencies into the target.
4. **MVP model:** prioritize the approximately 425 MB Qwen3.5-0.8B H128/Q4-G32-DOT4 artifact as the intended first performance model. Keep Qwen3-0.6B from `llm.asm` as a platform bring-up/reference option, not the product's primary model target.
5. **Performance goal:** maximize prompt prefill and decode throughput; 150+ generated tokens/sec is a stretch goal, not an assumed outcome of removing Linux. Correctness, repeatability and measured end-to-end gains are mandatory.

This decision preserves the existing BareMetalInfer scope: x86-64 AMD/Intel, CPU-only, inference-only, bounded memory, one sequence initially, QEMU/KVM development and no host OS dependency in the final image.

## 2. Compared projects

| Dimension | `alesha-pro/llm.asm` | `Danmoreng/qwen35-cpu` |
|---|---|---|
| Primary strength | Minimal x86-64 bootable environment and OS-independent execution | Qwen3.5-specific CPU inference, quantization and kernels |
| Implementation | NASM x86-64 assembly | C++20 |
| Runtime model | Linux static ELF and Multiboot bare-metal OS variants | Hosted native CPU engine/server |
| Model | Qwen3-0.6B, conventional decoder/GQA | Qwen3.5-0.8B and related dense model variants |
| Quantization | Q8_0, with optimized AVX2 INT8 path | H128/Q4-G32-DOT4 specialized format |
| SIMD | AVX2, FMA, F16C | AVX2 and AVX-512 dispatch; VNNI-oriented Q4 path |
| Attention/state | Conventional attention and contiguous KV cache | Hybrid Gated DeltaNet + full attention, recurrent/convolution state and KV |
| Parallelism | Linux clone/futex in hosted phase; APIC/SMP in bare-metal phase | Native worker/executor implementation |
| OS dependencies | Bare-metal variant has no Linux/libc runtime dependency | Uses hosted C++ facilities and is not directly freestanding |
| Best reuse | Boot, memory, low-level CPU, serial, APIC/SMP design patterns | Model format/quantization semantics, Qwen3.5 math, kernel/reference behavior |
| Main limitation for this project | Its Qwen3-0.6B inference path is not Qwen3.5-compatible | Its current runtime is not a `no_std` bare-metal runtime |

## 3. What can and cannot be reused

### 3.1 From `llm.asm`

**Reuse as design/reference material:**
- Small bootable x86-64 image and a clean loader-to-kernel handoff.
- Long-mode, page-table and early CPU initialization sequence.
- Explicit physical memory map and simple allocator approach.
- Serial console and early fault diagnostics.
- CPUID/SIMD initialization discipline.
- ACPI MADT/APIC discovery, application-processor startup and bounded SMP barriers.
- A shared inference/platform boundary so hosted and bare-metal execution can use the same model-level contracts.

**Do not copy blindly:**
- Its Qwen3-0.6B transformer implementation as the Qwen3.5 model adapter.
- Its Q8_0 AVX2 kernels as a substitute for H128/Q4-G32-DOT4.
- Its Multiboot1 boot details if the branch's chosen loader uses another protocol.
- Its published throughput as a direct comparison to this machine/model/workload.

The 24 KB figure describes the small OS/kernel image, not the full runtime memory footprint: model weights and runtime state are additional memory consumers.

### 3.2 From `qwen35-cpu`

**Use as the model-specific reference:**
- Qwen3.5-0.8B architecture and tensor/state interpretation.
- H128 transform and Q4-G32-DOT4 quantization layout.
- Quantized dot/matvec behavior and AVX2/AVX-512 dispatch concepts.
- Gated DeltaNet recurrence, convolution history and full-attention behavior.
- KV-cache and recurrent-state lifecycle requirements.
- Prefill/decode separation and numerical/token parity oracle.
- Model packer/metadata validation requirements derived from its artifact format.

**Do not import the hosted engine wholesale:**
- The existing runtime uses `std` facilities such as filesystem, streams, mutexes, random and hosted worker/thread primitives.
- Its C++ translation units and ownership model are not a freestanding `no_std` implementation.
- Do not add a C++ runtime, libc, OS threads, filesystem or hosted synchronization to the BareMetalInfer target merely to reuse code.
- Do not introduce a large generic abstraction layer solely to mirror the upstream C++ API.

Use its hosted executable/reference in development tests where possible. Port only the minimum required algorithms and kernels into Rust/intrinsics or narrow assembly after differential tests exist.

## 4. Why a direct merge is incorrect

The two projects solve different layers and target different model architectures.

- `llm.asm` implements Qwen3-0.6B's conventional attention path. Qwen3.5-0.8B has hybrid Gated DeltaNet and full-attention blocks, with recurrent state and convolution history. Replacing one with the other would produce incorrect model semantics.
- The quantization paths differ. Q8_0 AVX2 kernels cannot be treated as equivalent to H128/Q4-G32-DOT4 kernels.
- The bare-metal assembly project has platform primitives but not the Qwen3.5 architecture implementation.
- The Qwen3.5 C++ engine has model behavior and optimized kernels but assumes a hosted runtime.

Therefore, integrate at explicit boundaries: platform services from the BareMetalInfer runtime; model semantics and kernel requirements from Qwen3.5; a hosted oracle for parity. Do not merge repositories or share code across incompatible license/build/runtime assumptions without a source-level audit.

## 5. Selected target architecture

```text
Existing Ubuntu 26 host
  |
  +-- KVM/QEMU + hosted test tools
  |     |
  |     +-- qwen35-cpu reference/oracle (hosted)
  |     +-- Rust hosted test adapter
  |     +-- benchmark + differential-test harness
  |
  +-- QEMU/KVM launches freestanding BareMetalInfer image
        |
        +-- Existing standard bootloader
        |     (one protocol selected and validated)
        |
        +-- Rust x86-64 no_std kernel/runtime
        |     +-- BootInfo normalization
        |     +-- serial / exception diagnostics
        |     +-- bounded physical memory + arenas
        |     +-- CPUID + XCR0 feature detection
        |     +-- timer and CPU topology
        |     +-- APIC/SMP services (after single-core MVP)
        |
        +-- Model-neutral inference contracts
        |     +-- Tensor / quantization descriptors
        |     +-- ModelSource / tokenizer / sampler
        |     +-- prefill / decode lifecycle
        |     +-- bounded sequence state
        |
        +-- Qwen3.5-0.8B adapter
              +-- H128/Q4-G32-DOT4 weights
              +-- embedding / norm / projections / MLP
              +-- Gated DeltaNet + convolution state
              +-- full attention + KV cache
              +-- greedy sampling
              +-- scalar reference + optimized SIMD dispatch
```

### 5.1 Platform ports

Keep the inference core independent of boot protocol and hosted OS:

- `BootInfoProvider`: normalized boot data and memory map.
- `PhysicalMemory` / `Arena`: bounded allocation and accounting.
- `CpuInfo`: CPUID, XCR0 and topology.
- `CoreControl` / `WorkerPool`: single-core initially; APIC/SMP later.
- `Clock`: monotonic counter and calibrated frequency.
- `Console`: serial output initially.
- `ModelSource`: validated immutable model sections.
- Hosted implementations may use ordinary OS facilities, but those implementations must not be linked into the freestanding artifact.

Avoid virtual dispatch in per-element/per-token hot loops. Resolve kernel and platform choices at initialization or plan boundaries.

### 5.2 Model adapter requirements

The adapter must describe model semantics, not own physical memory policy. It must explicitly represent:
- Layer/block topology and which blocks use DeltaNet versus full attention.
- Tensor names, shapes, strides, quantization and alignment.
- DeltaNet recurrent state and convolution history.
- Attention KV geometry and cache lifecycle.
- Tokenizer assets, chat/template requirements and vocabulary.
- Prefill and decode execution plans.
- Supported context and memory requirements.

Do not assume Qwen3.5 behavior from a family label alone. Validate the exact model artifact and metadata before loading.

## 6. MVP and delivery sequence

### Stage 0 — Environment and reference baseline
- Keep Ubuntu 26 as the host; use the documented KVM/QEMU workflow.
- Build and run the hosted Qwen3.5-0.8B reference on the exact selected artifact.
- Record artifact hash, tokenizer, quantization, compiler, CPU/ISA, prompt corpus, context and output length.
- Preserve the existing local native benchmark report as a baseline; re-run it using the same harness before drawing comparisons.

### Stage 1 — Freestanding boot and platform skeleton
- Boot a minimal Rust `no_std` image through one existing loader.
- Validate boot info, memory map, serial output, exception path and static dependency audit.
- Keep Qwen3-0.6B assembly model out of the main model acceptance criteria; it may be used as a boot/platform smoke fixture only if it materially accelerates bring-up.

### Stage 2 — Qwen3.5 scalar correctness
- Define packed model format and read-only model source.
- Implement checked tensor/quantization descriptors.
- Port tokenizer and model semantics.
- Implement scalar reference operations and Qwen3.5 block behavior.
- Validate intermediate tensors/logits against the hosted reference.
- Keep one core, one sequence and bounded contiguous state.

### Stage 3 — Quantized and SIMD kernels
- Implement H128/Q4-G32-DOT4 scalar reference first.
- Add AVX2 and optional AVX-512/VNNI implementations behind safe runtime dispatch.
- Differential-test every optimized kernel.
- Benchmark prefill and decode separately; tune per kernel and per CPU.
- Retain scalar fallback.

### Stage 4 — Bare-metal end-to-end generation
- Load the selected ~425 MB model artifact from a boot module or validated memory-mapped source.
- Run prefill and greedy decode with bounded KV and DeltaNet state.
- Compare logits and greedy output against the hosted oracle.
- Record memory high-water, throughput and failure behavior.

### Stage 5 — Multicore and platform optimization
- Add APIC/SMP only after single-core correctness.
- Compare 1/2/4/8 worker counts and synchronization overhead.
- Tune affinity, memory layout, prefetch, cache reuse and kernel scheduling based on profiles.
- Consider assembly kernels only when Rust intrinsics cannot meet a measured target.

### Stage 6 — Additional boot path / model expansion
- Add direct UEFI/firmware boot only after the standard-loader image runs inference.
- Add other model families or advanced memory systems only when there is a demonstrated use case.

## 7. Performance objectives and evidence

The user's local native CPU report (Ryzen 7 9700X, 8 physical cores, AVX-512/VNNI, Qwen3.5-0.8B H128/Q4-G32-DOT4, 425 MB artifact) was previously reported as:

| Context | Prefill | Decode |
|---:|---:|---:|
| 512 | 2,830 tok/s | 108.40 tok/s |
| 4K | 2,327 tok/s | 89.76 tok/s |
| 8K | 1,889 tok/s | 77.11 tok/s |
| 16K | 1,372 tok/s | 59.21 tok/s |

These are **user-reported native Linux measurements**, not independently reproduced by this document update. Preserve the source report and its exact benchmark conditions. The 150+ tok/s decode goal is a stretch target; do not claim that bare-metal execution will automatically achieve it.

The `llm.asm` published figures use Qwen3-0.6B Q8_0 on different EPYC/Xeon systems and different thread/vCPU configurations. Its reported prefill comparison is not apples-to-apples because its prompt path differs from the batched-GEMM path used by llama.cpp. Do not compare these figures directly with the Ryzen 7 9700X/Qwen3.5 report.

### Required benchmark protocol
- Same model bytes and SHA-256, tokenizer and quantization.
- Same prompt token IDs, context lengths and generated token count.
- Greedy sampling for deterministic parity runs.
- Separate prefill, decode, time-to-first-token and end-to-end latency.
- Record CPU model, microcode, guest-visible ISA, XCR0, worker count, affinity, clock/frequency and thermal state.
- Label results as hosted Linux, freestanding QEMU/KVM or physical bare metal.
- Run warmups and repeated trials; report median and spread.
- Include peak memory, cache behavior and per-kernel timing where available.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Qwen3.5 hybrid architecture is more complex than the Qwen3 assembly model | Port model semantics from qwen35 reference; test each block/state path |
| Hosted qwen35-cpu cannot be linked directly into no_std | Keep it as oracle; implement freestanding Rust equivalents |
| AVX-512/VNNI may not always outperform AVX2 | Runtime dispatch and per-kernel benchmarking; preserve scalar fallback |
| Bare metal may not improve throughput materially | Measure kernel, memory and synchronization costs; do not assume OS overhead dominates |
| Model packer or artifact changes break tensor offsets | Versioned format, checksums, bounds/overlap/alignment validation |
| QEMU results differ from physical hardware | Separate benchmark classes and require physical validation before claims |
| Scope expands into OS development | Use existing bootloader; implement only inference-required platform services |

## 9. Architecture invariants

1. No OS, libc, POSIX, GPU or vendor runtime dependency in the freestanding target.
2. Rust `no_std` is the primary implementation; assembly remains a narrow, measured boundary.
3. The hosted reference is a correctness oracle, not a runtime dependency.
4. Qwen3.5 model semantics are not substituted with Qwen3 semantics.
5. Scalar kernels define correctness; SIMD implementations must pass differential tests.
6. All memory is bounded, validated and accounted.
7. Single-core correctness precedes SMP.
8. One standard boot path precedes direct firmware boot.
9. No performance claim without matched, reproducible workloads.

## 10. Upstream references

- [alesha-pro/llm.asm](https://github.com/alesha-pro/llm.asm) — x86-64 assembly inference and Multiboot bare-metal OS.
- [Danmoreng/qwen35-cpu](https://github.com/Danmoreng/qwen35-cpu) — Qwen3.5 CPU inference and H128/Q4-G32-DOT4 implementation.
- [BareMetalInfer architecture](BAREMETAL_INFER_ARCHITECTURE.md)
- [Platform and language strategy](PLATFORM_AND_LANGUAGE_STRATEGY.md)
- [Implementation roadmap](IMPLEMENTATION_ROADMAP.md)
- [Implementation work packages](IMPLEMENTATION_WORK_PACKAGES.md)
- [Development VM setup](DEVELOPMENT_VM_SETUP.md)
