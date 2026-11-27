# BareMetalInfer — Reference Architecture

**Status:** Decision baseline  
**Branch:** `BareMetalInfer`  
**Scope:** x86-64 bare-metal CPU inference only.

## 1. Product definition

BareMetalInfer is a small, OS-independent LLM inference engine that boots on AMD and Intel x86-64 systems and executes transformer inference directly on CPU cores.

The target runtime has no dependency on Linux, Windows, macOS, POSIX, libc, GPU, GPU driver, vendor compute SDK, or vendor runtime. Firmware and a boot mechanism are still required; "bare metal" means no host operating system is required, not that firmware is absent.

The first release is inference-only. It does not implement a general-purpose OS, training, networking, filesystems, GPU execution or multi-architecture runtime.

## 2. Locked decisions

| Decision | Adopted direction |
|---|---|
| Boot | Support both a standard bootloader path and a future native/direct boot path |
| CPU | x86-64 only initially; AMD and Intel |
| Model | Small dense decoder-only model first |
| Memory | Simple bounded physical-memory layout and fixed arenas; no advanced paging/paged-KV subsystem initially |
| Scope | Inference only; implement only platform services inference requires |
| Primary language | Rust `no_std` |
| Low-level code | Small x86-64 assembly boundary and measured SIMD kernels |

The standard-loader path is the first implementation path. The direct/native boot path is a second boot adapter, not a prerequisite for completing inference. Both must enter the same kernel/runtime entry contract.

## 3. Layered architecture

```text
             Firmware
                |
      +---------+----------+
      |                    |
 Standard boot path   Native boot path
 (Multiboot2/UEFI)    (later milestone)
      |                    |
      +---------+----------+
                |
          x86 Boot Adapter
                |
       Minimal CPU/Memory Init
                |
        Bare-metal Runtime
 (allocator, AP startup, timer, console)
                |
          Inference Core
                |
       Model Adapter (small dense)
                |
        CPU Kernel Dispatcher
                |
      Scalar / x86 SIMD Kernels
                |
       Model + Tokenizer + KV
```

Boot adapters must provide the same validated `BootInfo` contract: memory map, kernel/module ranges, framebuffer/serial information when available, firmware tables and boot method. Inference code must never inspect boot protocol structures.

## 4. Runtime boundaries and interfaces

Keep interfaces small and capability-oriented:

- `BootInfoProvider`: validated memory map, modules and boot metadata.
- `PhysicalMemory`: allocate/free bounded page ranges and reserve regions.
- `Arena`: aligned allocation from fixed pre-reserved regions.
- `CpuInfo`: topology and runtime ISA feature discovery.
- `CoreControl`: start secondary cores and expose core identifiers.
- `WorkerPool`: bounded worker execution, barriers and wake/wait.
- `Clock`: monotonic ticks and calibrated frequency.
- `Console`: serial output initially; framebuffer/keyboard optional.
- `ModelSource`: immutable byte ranges for packed model assets.

Do not create a broad generic HAL that combines unrelated capabilities. Do not expose physical addresses outside memory/platform modules.

## 5. Memory design — deliberately simple

Use a validated firmware/bootloader memory map and a small number of fixed regions. The first implementation uses a bump/arena allocator with explicit alignment, bounds and high-water accounting. Add a small free-list/page allocator only where boot/runtime needs reusable pages.

Initial regions:

1. Kernel image, page tables and per-core stacks.
2. Read-only model and tokenizer module.
3. KV-cache arena.
4. Activation and scratch arena.
5. Runtime metadata and worker state.
6. Reserved firmware and unavailable memory.

Use only the minimum page tables required to enter and operate in x86-64 long mode. Do not build a general virtual-memory manager, swap, demand paging, NUMA policy or huge-page optimizer in the first milestone. Huge pages are an optional later experiment, not an architectural dependency.

KV-cache starts as one bounded contiguous allocation per active sequence. No paged KV, eviction, CPU/GPU tiering or streaming subsystem in the initial scope. Define a small KV interface so its implementation can evolve without exposing physical addresses.

Model weights should be consumed from a validated read-only boot module where practical, avoiding unnecessary copies. The packer must align tensor sections and validate all offsets and lengths before the target accesses them.

## 6. Inference core

The inference core is `no_std` and independent of boot protocol, firmware and CPU vendor. It owns:

- Tensor metadata and shape validation.
- Dtype and quantization descriptors.
- Small dense decoder model adapter.
- Tokenizer contract and sampling.
- Prefill/decode lifecycle.
- Contiguous KV-cache contract.
- Scalar reference operations.
- Kernel dispatch contract.
- Memory and performance counters.

Do not build a generic plugin framework or support multiple model families before one end-to-end model is correct.

## 7. CPU execution

Start with a scalar Rust reference backend. Add x86-64 variants only after correctness and profiling:

- x86-64 baseline.
- SSE2 where useful for baseline-compatible vectorization.
- AVX2 + FMA.
- Optional AVX-512 after hardware coverage and benchmark evidence.

Use runtime CPUID and OS-independent architectural-state checks before dispatching extended ISA instructions. Never infer instruction availability from AMD/Intel vendor strings. Keep scalar fallback available.

Separate prefill and decode kernels. Prioritize matrix-vector/decode and matrix-matrix/prefill independently. Add fused QKV or gate/up only when the selected model architecture supports the fusion and numerical tests prove equivalence.

## 8. Boot strategy

### Path A — standard loader (first)

Use a documented existing bootloader/protocol to load the kernel and model module. The bootloader is a build/boot-time dependency, not a runtime OS dependency. The kernel validates every address, size, alignment and memory-map entry before use.

### Path B — native/direct boot (later)

Implement a small x86 firmware entry path only after Path A has a working inference image. Keep firmware entry, mode transition and boot metadata conversion in a separate adapter. Both paths must produce the same `BootInfo`.

## 9. Verification and benchmarks

Every optimized kernel has a scalar reference and differential tests for absolute/relative error, NaN/Inf, logits and greedy-token agreement. Test single-core before multicore. Benchmark prefill, decode, time-to-first-token, memory footprint and scaling separately. Record CPU model, microcode/firmware where available, ISA, clock policy, memory, model hash, quantization, prompt/context and worker count.

## 10. Non-goals

- No GPU, GPU driver, GPU compute API or vendor SDK.
- No OS runtime, libc, POSIX, syscalls or hosted runtime dependency.
- No ARM64 in this project baseline.
- No general-purpose OS, filesystem, network stack or shell.
- No training or fine-tuning.
- No advanced virtual-memory manager, paged KV, swap or GPU/CPU tiering.
- No multi-model server or general plugin ecosystem in the first release.

## 11. Architectural invariants

1. Both boot paths enter the same kernel contract.
2. Inference core never depends on boot protocol or firmware.
3. Scalar kernels define reference behavior.
4. ISA dispatch is runtime-checked and always has a safe fallback.
5. Memory is bounded, aligned, accounted and validated.
6. Physical addresses do not cross into model/inference APIs.
7. The smallest working model is delivered before framework generalization.
8. No optimization is accepted without reproducible end-to-end measurements.

## 12. Development and validation environment

Development must not require changing the developer's existing host OS or boot configuration. The reference development environment is an Ubuntu 26.04 host using KVM, QEMU and libvirt, with an isolated Ubuntu Server development VM. See [DEVELOPMENT_VM_SETUP.md](DEVELOPMENT_VM_SETUP.md) for the reproducible setup.

Use two execution modes:
- Hosted Linux backend in the development VM for rapid development, reference comparisons and kernel benchmarking.
- Freestanding BareMetalInfer image launched directly by host QEMU/KVM, with no guest Linux OS, for boot and runtime validation.

The Linux backend is a development adapter, not a dependency of the freestanding inference core. Keep OS-specific allocation, timing, threading and console implementations behind narrow platform interfaces. The freestanding target must continue to build as `no_std` with no libc, syscall, POSIX or Linux runtime dependency.

Initial VM profile: host CPU passthrough, 6 vCPUs for development and 8 vCPUs for dedicated benchmarks, 24–32 GiB guest RAM, 100 GiB QCOW2 disk, Q35 and OVMF. Tune vCPU pinning only after inspecting actual host CPU topology. Keep host resources available and do not assume VM CPU pinning makes execution physically bare metal.

All reports must distinguish Linux-guest, freestanding-QEMU/KVM and physical bare-metal results. QEMU/KVM validates the freestanding guest and CPU feature paths but does not provide direct physical motherboard, memory-controller, firmware or interrupt-controller access.
