# BareMetalInfer — Reference Architecture

**Status:** Proposed architecture baseline  
**Branch:** `BareMetalInfer`  
**Purpose:** CPU-only, operating-system-independent LLM inference on supported bare-metal systems.

## 1. Product definition

BareMetalInfer is a standalone inference system that boots on supported hardware and runs transformer inference directly on CPU resources. It does not require a host operating system, GPU, GPU driver, vendor compute SDK, vendor runtime, libc, or a general-purpose userspace.

The project is CPU-native. It is not a GPU inference runtime and does not inherit the ROCm/HIP or CUDA architecture from the original Infer project.

The inference core must be independent of the boot mechanism and platform initialization. Platform-specific code is isolated behind narrow interfaces.

## 2. Supported processor families

Initial architecture families:

- x86-64: AMD and Intel processors.
- AArch64: ARMv8-A and later, with Apple Silicon treated as a separate platform feasibility target.

Runtime CPU feature detection is mandatory. Never assume AVX2, FMA, AVX-512, NEON, or SVE merely from the processor family. Select only kernels supported by the detected CPU and enabled OS-independent architectural state.

Important distinction: CPU ISA portability does not guarantee firmware/boot portability. Apple Silicon has a platform-specific boot and security chain. The portable inference core can target AArch64, but generic bare-metal boot on Apple hardware is not promised until independently validated.

## 3. Layered architecture

```text
       Firmware / Boot Protocol
                 |
          Boot & Platform HAL
                 |
       CPU / Memory Initialization
                 |
        BareMetal Runtime Services
       (alloc, threads, timers, I/O)
                 |
          Inference Application
                 |
       Scheduler / Request Runtime
                 |
        Model Architecture Adapter
                 |
       CPU Execution / Kernel Planner
                 |
       Tensor & Quantization Kernels
                 |
       Model + Tokenizer + KV State
```

### 3.1 Boot and platform HAL

Owns entry point, CPU mode setup, interrupt/exception tables, physical memory map, page tables, AP startup, timers and minimal console/input. Boot protocol and firmware details must not leak into inference crates.

### 3.2 Runtime services

Provide small OS-independent contracts for:

- Physical and virtual memory allocation.
- Memory mapping and model loading.
- CPU topology and feature discovery.
- Worker creation, barriers, atomics and wait/wake.
- Monotonic time and profiling.
- Console, serial and optional framebuffer output.
- Panic, exception and fault reporting.

No POSIX, pthread, Linux syscall, Windows API, libc or hosted async runtime dependency.

### 3.3 Inference core

Owns model-neutral tensor metadata, quantization descriptors, tokenizer interface, model adapter registry, prefill/decode lifecycle, sampling, KV cache, scheduling, telemetry and resource accounting.

The inference core must compile with `no_std` and `alloc`. Hardware-specific operations are accessed through small traits.

### 3.4 Model adapters

Keep model topology separate from kernels and platform services. Start with a small dense decoder-only transformer adapter, then add Qwen-family support after tensor mapping, normalization, RoPE, attention, MLP and tokenizer contracts are validated.

Do not assume Qwen3.8 compatibility until the exact architecture and checkpoint format are verified.

### 3.5 CPU execution backend

Provide a scalar correctness backend and optimized kernel sets:

- Generic scalar Rust reference.
- x86-64 baseline SIMD.
- x86-64 AVX2 + FMA.
- Optional AVX-512 variants.
- AArch64 NEON.
- Optional SVE/SVE2 variants.

Kernel selection uses runtime feature detection plus shape, dtype and workload characteristics. Keep prefill and decode plans separate.

## 4. Memory architecture

Use explicit physical-memory and virtual-memory abstractions. Separate logical model tensors and KV blocks from their physical addresses.

Initial memory regions:

- Kernel/runtime region.
- Model weights region.
- Tokenizer region.
- KV-cache region.
- Activation/workspace arena.
- Per-core stacks and scheduler metadata.
- Reserved firmware and device memory.

Support aligned arenas, bounded allocation, memory-map validation, huge pages where available, and explicit ownership. Model loading should avoid unnecessary copies: a boot module or memory-mapped image may be registered as a read-only model region.

KV-cache is a replaceable subsystem. Begin with contiguous per-sequence storage; introduce block/paged allocation only after correctness and fragmentation measurements justify it. No GPU/CPU tiering is required because this project is CPU-only, but the logical KV API must not expose physical addresses.

## 5. Threading and CPU topology

Use a platform-neutral worker interface. Platform implementations may use APIC/IPI on x86-64 and architecture-appropriate interrupt/mailbox mechanisms on ARM64. The inference scheduler must not know those details.

Start with static worker pools and deterministic barriers. Add work stealing only if benchmarks justify it. Bound all queues and make shutdown/resource reclamation explicit.

## 6. Numerical and verification architecture

Every optimized kernel must have a scalar reference implementation and differential tests. Verify:

- Absolute and relative error.
- Cosine similarity where appropriate.
- NaN/Inf absence.
- Logit agreement.
- Greedy token agreement over longer sequences.
- Thread-count determinism where the algorithm promises it.

Benchmark prefill and decode independently, recording CPU model, ISA features, clocks where available, memory configuration, model revision, quantization, context and worker count.

## 7. Non-goals

- No GPU execution or GPU driver support.
- No CUDA, ROCm, HIP, Metal, oneAPI or vendor compute SDK.
- No dependency on Linux, Windows, macOS, POSIX or libc at runtime.
- No hosted service, HTTP server or general-purpose userspace in the initial product.
- No training/fine-tuning in the initial inference milestone.
- No claim of generic bare-metal boot support on Apple Silicon before platform validation.

## 8. Architectural invariants

1. Inference core is independent of boot protocol, firmware and platform.
2. Scalar reference behavior is the correctness authority.
3. Runtime feature detection gates every specialized kernel.
4. No raw physical address crosses model/application interfaces.
5. All allocations are bounded, accounted and reclaimable.
6. Platform-specific code is isolated behind HAL/runtime contracts.
7. Optimize only against reproducible end-to-end measurements.
8. Prefer the smallest implementation that meets the current milestone.
