# BareMetalInfer — Development VM and Validation Environment

**Host:** Existing Ubuntu 26.04 installation  
**Virtualization:** KVM + QEMU + libvirt  
**Target:** AMD/Intel x86-64 CPU inference; no GPU dependency  
**Purpose:** Isolated development and repeatable virtual-hardware validation without changing the host OS or boot configuration.

## 1. Environment decisions

Use the existing Ubuntu host as the hypervisor. Do not replace the host OS, repartition disks, install a second host OS, or change the host bootloader.

Use two execution modes:

1. **Linux development/benchmark VM:** Ubuntu Server guest for compiling, testing and running the hosted platform adapter. This provides a convenient development environment and a baseline for inference-kernel performance.
2. **Freestanding test VM:** Launch the Limine-based BareMetalInfer image directly with QEMU/KVM from the Ubuntu host. The guest contains no Linux OS; it boots the BareMetalInfer image.

Do not initially nest QEMU inside the development VM. Host-launched QEMU avoids nested-virtualization complexity and provides a more direct view of the host CPU features.

The Linux VM is a development and comparison environment. It is not the final runtime architecture and must not cause Linux APIs or libc to leak into the freestanding target.

## 2. Initial resource allocation

Host hardware baseline: Ryzen 7 9700X (8 physical cores / 16 logical processors), 64 GB RAM and 1 TB NVMe.

| Resource | Development VM | Benchmark configuration |
|---|---:|---:|
| vCPU | 6 | 8 |
| Guest RAM | 24–32 GB | 32 GB |
| Virtual disk | 100 GB QCOW2 | Same |
| Firmware | OVMF UEFI | OVMF UEFI |
| Machine | Q35 | Q35 |
| CPU model | host-passthrough | host-passthrough |
| Network | libvirt NAT | Disabled or isolated where practical |
| GPU | None | None |

Start with 6 vCPUs and 24 GB RAM if the host is used interactively. Increase to 8 vCPUs and 32 GB for dedicated benchmark sessions. Do not reserve all host resources permanently.

The VM allocation is a ceiling, not a guarantee of exclusive physical resources. Keep sufficient memory and CPU capacity for the host.

## 3. Host packages and checks

Install the distribution packages:

```bash
sudo apt update
sudo apt install -y \
  qemu-system-x86 qemu-utils \
  libvirt-daemon-system libvirt-clients \
  virtinst virt-manager virt-viewer \
  ovmf cpu-checker bridge-utils
```

Verify hardware virtualization and KVM:

```bash
kvm-ok
ls -l /dev/kvm
lsmod | grep kvm
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
```

On AMD, verify the SVM virtualization feature is available. If KVM is unavailable, inspect firmware virtualization settings and host kernel logs before proceeding.

## 4. VM creation

Use virt-manager for initial installation and inspection.

Recommended VM settings:

- Name: `BareMetalInfer-Dev`
- Guest: Ubuntu Server (use a currently supported LTS release available in the installer)
- Firmware: OVMF UEFI
- Machine type: Q35
- CPU mode: host-passthrough
- vCPU: 6 initially; 8 for benchmark runs
- RAM: 24–32 GB
- Disk: 100 GB QCOW2 on the NVMe
- Network: libvirt NAT for package installation; optionally disconnect during benchmark runs
- Display: VNC/SPICE during setup; serial console/SSH for routine use

Use a normal virtual disk initially. Direct NVMe passthrough is unnecessary for model files of this size and adds host-storage risk.

## 5. CPU topology, pinning and frequency

Always inspect the real topology with `lscpu -e`; do not assume logical CPU numbering or SMT sibling placement.

For controlled benchmarks:

- Pin each guest vCPU to a distinct physical core's logical CPU where possible.
- Leave sibling threads and some host capacity available during interactive development.
- Pin QEMU emulator threads separately only after measuring their impact.
- Avoid CPU overcommit and competing host workloads.
- Record the guest vCPU topology and host CPU mapping with every benchmark.

Pinning improves repeatability but does not make a VM equivalent to physical bare metal. Host scheduling, firmware power management, interrupt virtualization and memory virtualization remain relevant.

Do not change the host CPU governor or power settings automatically. Record the active policy and use a consistent, thermally stable configuration for comparisons.

## 6. Verify guest CPU features

Inside the Linux guest:

```bash
lscpu
grep -m1 '^flags' /proc/cpuinfo
```

Check for AVX2, FMA and the AVX-512 subsets required by each optimized kernel. Feature visibility is not sufficient by itself: BareMetalInfer must perform CPUID leaf checks and validate OSXSAVE/XCR0 state before AVX-family dispatch.

Do not hard-code the expectation that AVX-512 is exposed in every VM configuration. Treat the guest-visible feature set as authoritative and preserve a scalar fallback.

## 7. Guest development toolchain

Install only what is needed for development and host-side tests:

```bash
sudo apt update
sudo apt install -y \
  build-essential clang lld llvm nasm \
  cmake ninja-build git gdb \
  python3 python3-venv python3-pip \
  qemu-system-x86 qemu-utils \
  xorriso mtools openssh-server
```

Install and pin the Rust toolchain with rustup and commit the project's toolchain configuration. Host-side packers and reference tests may use standard libraries and filesystem access. The freestanding target remains `no_std` and must not link host tools or OS libraries.

## 8. Freestanding QEMU/KVM test path

Build the Limine boot image inside the development workflow, but initially launch the freestanding image from the Ubuntu host. Example command (adapt firmware, disk/image arguments and serial configuration to the actual build output):

```bash
qemu-system-x86_64 \
  -enable-kvm \
  -machine q35,accel=kvm \
  -cpu host \
  -smp 8,sockets=1,cores=8,threads=1 \
  -m 8G \
  -drive file=baremetalinfer.img,format=raw \
  -serial stdio \
  -display none
```

This is a virtual machine booting a freestanding guest image. It validates boot protocol handling, memory-map parsing, CPU feature discovery and inference behavior without a guest OS. It does not provide direct physical memory-controller, motherboard firmware, APIC or device access.

The exact QEMU machine, firmware and image options must match the boot artifact. Do not treat the illustrative command as a validated build script until integrated and tested.

Nested QEMU inside the Linux guest is deferred. Enable nested AMD virtualization only if a later workflow demonstrates a concrete need.

## 9. Benchmark policy

Run hosted Linux and freestanding guest benchmarks with the same:

- Model artifact and SHA-256.
- Quantization and tokenizer.
- Prompt corpus, token counts and context lengths.
- Sampling configuration (greedy for deterministic comparisons).
- Kernel implementation and compiler settings where applicable.
- Worker count and guest-visible ISA.
- Warm-up, repetitions and measurement methodology.

Record at minimum:

- Host CPU, microcode, BIOS/firmware and kernel.
- QEMU/KVM/libvirt versions and VM XML/configuration.
- Guest CPU flags and vCPU pinning.
- Model hash, quantization and context length.
- Prefill tokens/sec, decode tokens/sec and time-to-first-token.
- Peak memory, CPU utilization/frequency and run-to-run variance.

Keep Linux VM, freestanding QEMU/KVM and physical bare-metal results in separate result categories. Never label a virtualized result as physical bare-metal performance. A 150+ tokens/sec target is an engineering objective, not a VM-guaranteed outcome.

## 10. Safety and operational constraints

- Keep the host OS and boot configuration unchanged.
- Store VM images in a dedicated directory with sufficient free space.
- Use QCOW2 snapshots before risky guest changes.
- Do not pass through the host NVMe controller or GPU for this CPU-only project.
- Do not run untrusted boot images with host device passthrough.
- Avoid hugepages until baseline performance is measured; add them only as a controlled experiment.
- Do not dedicate all host RAM or cores permanently to the VM.

## 11. Environment acceptance checklist

- [ ] KVM works on the Ubuntu host.
- [ ] Development VM boots and is accessible over SSH.
- [ ] Guest sees expected CPU features and vCPU topology.
- [ ] Guest toolchain builds the hosted inference target.
- [ ] Host QEMU/KVM boots the freestanding image with serial output.
- [ ] Scalar and SIMD tests pass in the Linux guest.
- [ ] Freestanding QEMU smoke tests pass.
- [ ] Benchmark reports distinguish hosted VM from freestanding VM results.
- [ ] Host OS, bootloader and disk layout remain unchanged.
