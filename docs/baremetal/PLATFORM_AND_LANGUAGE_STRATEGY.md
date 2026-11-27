# BareMetalInfer — Language, Toolchain and Boot Strategy

## 1. Locked platform scope

Initial target: x86-64 AMD and Intel CPUs only. Do not add ARM64, Apple Silicon, GPU, vendor compute SDK or OS runtime dependencies to this branch's initial implementation.

AMD and Intel are both supported through architectural interfaces and runtime CPUID feature detection. Avoid vendor-specific code paths unless a measured, documented microarchitecture difference requires one.

## 2. Language combination

### Rust — primary language

Use stable Rust with `#![no_std]` for the kernel, runtime services, inference core, model adapter, tokenizer integration, tensor metadata, scalar kernels, quantization, scheduler and memory arenas.

Use `core` initially. Introduce `alloc` only after a bounded allocator is implemented and tested. Avoid crates that pull in `std`, OS threads, filesystem, sockets, dynamic loading, host clocks or hosted async runtimes.

### x86-64 assembly — narrow boundary

Use NASM for boot/entry stubs if it best fits the selected image format, and Rust `global_asm!`/LLVM assembly where it reduces toolchain complexity. Assembly is restricted to entry/transition code, interrupt stubs, secondary-core startup and kernels proven to need hand-written instructions.

### Rust intrinsics — preferred SIMD implementation

Prefer `core::arch::x86_64` intrinsics for SSE2, AVX2, FMA and optional AVX-512. Use assembly only when benchmarks show a material benefit or intrinsics cannot express a required sequence.

### C — not part of the planned runtime

Do not add C or C++ as a second implementation language. A tiny C ABI shim requires an explicit architecture decision and must not introduce libc or a vendor runtime.

## 3. Toolchain

- Rust stable with explicit `x86_64-unknown-none` target.
- Cargo workspace and committed lockfile for reproducibility.
- LLVM/LLD or the Rust-supported linker path, with a reviewed linker script.
- NASM only for the small boot assembly boundary.
- QEMU x86-64 for automated boot tests.
- OVMF/UEFI firmware and a Multiboot2-capable loader as applicable to the chosen boot path.
- Host-side Rust/Python tools are permitted for model conversion, packing, image creation and test-data generation. They are build tools, not target runtime dependencies.

Audit the final ELF and boot image for undefined symbols, dynamic sections, relocations that cannot be resolved at load time, unexpected runtime crates and forbidden imports.

## 4. Two boot paths

### Path A: standard bootloader — implement first

Choose one documented protocol as the initial vertical slice (prefer Multiboot2 if the chosen loader and QEMU setup make it simplest). Keep a UEFI path as an alternative adapter if required for modern hardware coverage. Do not implement both loader protocols in the first sprint.

The loader supplies kernel/module addresses and a memory map. Validate all values and normalize them into the internal `BootInfo` structure. The loader is not an OS dependency; it is a boot-time component.

### Path B: native/direct firmware boot — implement second

After the standard-loader image boots and runs inference, add a direct x86 firmware entry path (UEFI application or another explicitly selected firmware interface). Keep firmware protocol calls in the boot adapter and convert them to the same `BootInfo` contract before transferring control to the shared kernel entry.

"Both" means both boot paths eventually exist; it does not mean a custom BIOS/UEFI bootloader must be written before inference works.

## 5. CPU feature detection and dispatch

- Build a conservative x86-64 baseline.
- Detect CPUID feature bits at runtime.
- Check OSXSAVE and XCR0 state before using AVX-family instructions.
- Keep scalar fallback and test dispatch with feature masks.
- Do not use AVX2/AVX-512 in generic code paths.
- Treat AVX-512 as optional and benchmark per CPU.
- Use vendor identity only for diagnostics or a proven tuning exception, not as the feature gate.

## 6. Build and test separation

Target runtime: `no_std`, no libc, no OS imports, no GPU/driver/SDK.

Host tools: may use `std`, Python and ordinary filesystem/process APIs. Keep them in a separate workspace package and never link them into the target image.

QEMU validates boot, memory contracts and basic inference behavior. Physical AMD and Intel hardware validation is required before declaring platform support.

## 7. Development VM and execution environments

The developer's existing Ubuntu 26.04 installation remains the host OS. Use KVM + QEMU + libvirt; do not replace the host OS, alter its bootloader or repartition its disks.

Maintain two distinct test paths:

1. **Linux development VM:** Ubuntu Server guest, used for compiling, hosted reference execution, model tooling and performance-kernel development.
2. **Freestanding QEMU/KVM guest:** launch the Limine-based BareMetalInfer image directly from the Ubuntu host. The guest boots BareMetalInfer without Linux.

Do not nest QEMU inside the development VM initially. Host-launched QEMU avoids requiring nested SVM and simplifies CPU feature exposure.

Recommended starting profile:
- Q35 machine type and OVMF UEFI firmware.
- CPU mode `host-passthrough`.
- 6 vCPUs for interactive development; 8 for dedicated benchmarks.
- 24–32 GiB RAM and 100 GiB QCOW2 disk.
- NAT networking for setup; disable network activity during controlled benchmark runs.
- Inspect actual topology before applying vCPU pinning; leave host capacity available.

Install the toolchain in the guest, pin Rust via rustup/toolchain configuration, and keep host-side packer/reference tools separate from the `no_std` target.

CPU feature exposure must be checked inside each guest. Runtime dispatch still requires CPUID bounds checks and OSXSAVE/XCR0 validation before AVX-family instructions. Do not assume host-passthrough exposes every host feature under all hypervisor configurations.

QEMU/KVM is suitable for boot, memory, exception, SIMD correctness and inference integration tests. It is not physical bare-metal validation: guest scheduling, interrupts, firmware, cache and memory behavior are virtualized. Keep three benchmark labels distinct: hosted Linux VM, freestanding QEMU/KVM guest and physical bare metal.

See [DEVELOPMENT_VM_SETUP.md](DEVELOPMENT_VM_SETUP.md) for installation, configuration, benchmark policy and acceptance checklist.
