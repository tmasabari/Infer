# BareMetalInfer — Language and Platform Strategy

## 1. Language selection

### Rust: primary implementation language

Use stable Rust with `#![no_std]` for the inference core, memory manager, scheduler, model adapters, tokenizer integration, quantization, scalar kernels and most platform services.

Use `alloc` only after the project provides a bounded global allocator. Avoid dependencies that silently require std, OS threads, filesystems, sockets, dynamic loading or host clocks.

Use Rust traits for narrow hardware/runtime boundaries. Avoid excessive crate fragmentation and avoid a universal HAL trait with unrelated methods.

### Assembly: narrow hardware and kernel boundary

Use architecture-specific assembly only for:

- Entry/boot transition and CPU mode setup.
- Interrupt/exception entry stubs and secondary-core startup.
- Context switch or low-level worker primitives if required.
- Measured hot kernels that cannot be expressed efficiently through intrinsics.

Use NASM or GNU assembler for x86-64 according to boot/toolchain integration. Use AArch64 GNU/LLVM assembler for ARM64. Keep assembly ABI wrappers small and documented.

### Rust architecture intrinsics

Prefer `core::arch` intrinsics for AVX2, FMA, AVX-512, NEON and SVE where supported by the compiler. This gives better type checking and easier testing than hand-writing every vector instruction.

### C: exception, not foundation

Do not introduce C as a parallel application language. Permit small C ABI glue only if a selected firmware, bootloader or pre-existing component has a stable C interface that is materially simpler and does not introduce an OS/vendor runtime dependency.

## 2. Toolchain

Preferred toolchain:

- Rust stable toolchain with explicit target triples and `no_std` targets.
- LLVM/LLD for linking and cross-target support.
- NASM for x86-64 assembly where useful.
- GNU/LLVM assembler for AArch64.
- Cargo workspace with reproducible locked dependencies.
- QEMU for x86-64 and AArch64 emulation; hardware validation remains mandatory.

Build host tools may use Python or Rust std. The booted target must not depend on the host toolchain or a host OS.

## 3. Boot strategy

Keep boot protocol replaceable. Start with a mature documented boot protocol/loader for the first vertical slice rather than writing a custom bootloader before inference works. A Multiboot2-compatible loader or a UEFI application path can be evaluated for x86-64. The loader supplies a memory map and kernel/module payloads; BareMetalInfer validates all ranges before use.

A later direct-boot path may remove the external bootloader where there is a clear benefit. Firmware remains unavoidable on commodity systems; "no OS dependency" does not mean "no firmware or boot protocol".

ARM64 boot must be evaluated separately. Do not assume x86 Multiboot, APIC, BIOS or PC memory-map behavior applies to ARM.

## 4. CPU feature policy

- Build a conservative baseline image.
- Detect CPU vendor/family/model and architectural features at runtime.
- Enable a feature only when hardware support and required architectural state are confirmed.
- Keep dispatch table-driven and test unsupported-feature fallbacks.
- Never require an AMD, Intel or Apple proprietary compiler/runtime.
- Do not branch on vendor identity when an architectural capability check is sufficient.

## 5. Portability stages

1. x86-64 QEMU reference platform.
2. AMD x86-64 physical hardware.
3. Intel x86-64 physical hardware.
4. AArch64 QEMU reference platform.
5. Generic ARM64 hardware with a documented boot protocol.
6. Apple Silicon boot feasibility investigation and separately scoped platform work.

The portable Rust inference core should be tested across targets before every boot platform is available.
