# BareMetalInfer — Implementation-Ready Work Packages

Target branch: BareMetalInfer  
Baseline: x86-64 AMD/Intel; Rust no_std; standard bootloader first and direct firmware boot second; small dense decoder model; simple bounded memory; inference only.

Paths below are proposed. Keep the workspace compact; do not create empty crates merely to match this document.

## WP-01 — Reproducible x86-64 no_std kernel skeleton

**Goal:** Produce the smallest bootable x86-64 image through one standard boot protocol.

**Interfaces**
- kernel_entry(boot_info_ptr): single Rust entry after loader handoff.
- BootInfo: versioned normalized structure containing memory-map entries, module ranges, framebuffer/serial metadata and boot method.
- BootMethod: identifies Multiboot2 initially; reserve a UEFI/direct value without implementing it yet.

**File-level changes**
- Cargo.toml, Cargo.lock: minimal workspace and pinned dependencies.
- .cargo/config.toml: x86_64-unknown-none target.
- linker/x86_64.ld: kernel layout and entry symbols.
- boot/x86_64/entry.asm: minimal entry/stack setup if required.
- src/main.rs: no_std, no_main, entry and panic handler.
- src/boot/{mod.rs,info.rs,multiboot2.rs}: normalize and validate boot data.
- scripts/build-image.sh and tests/qemu/boot-smoke.sh.

**Dependencies:** toolchain, QEMU and selected standard loader.

**Acceptance tests**
1. Clean checkout builds reproducibly with documented commands.
2. QEMU boots and emits BAREMETALINFER:BOOT_OK over serial.
3. Invalid/null/truncated boot-info fails safely.
4. Kernel ELF has no dynamic section, undefined hosted symbols or libc dependency.
5. No OS syscalls, GPU APIs or vendor SDK symbols are linked.

## WP-02 — Serial diagnostics, panic and CPU exceptions

**Goal:** Make early failures diagnosable without an OS.

**Interfaces**
- Console::write(bytes).
- PanicReporter::report(info).
- Exception entry stubs dispatch to a minimal Rust fault reporter.

**File-level changes**
- src/platform/x86_64/serial.rs
- src/platform/x86_64/idt.rs
- src/platform/x86_64/gdt.rs
- src/diagnostics/{panic.rs,log.rs}
- boot/x86_64/interrupts.asm
- tests/qemu exception tests.

**Dependencies:** WP-01.

**Acceptance tests**
1. Serial output works without firmware console services.
2. Deliberate breakpoint and invalid-opcode tests report vector/error context and halt.
3. Panic path does not allocate or unwind.
4. Interrupt handlers do not call inference code or use uninitialized runtime state.

## WP-03 — Simple physical memory map and arenas

**Goal:** Implement the selected simple memory strategy; avoid a general-purpose VM.

**Interfaces**
- MemoryRegion: start, length and kind.
- PhysicalMemoryMap: usable regions.
- Arena::alloc(layout): aligned bounded allocation.
- MemoryStats: capacity, used and high-water mark.

**File-level changes**
- src/memory/{mod.rs,region.rs,map.rs,arena.rs,layout.rs}
- src/boot/memory_map.rs
- src/model/region.rs for immutable model module views.
- tests/memory/{arena.rs,map.rs}

**Dependencies:** WP-01.

**Acceptance tests**
1. Reserved firmware, kernel, module and MMIO ranges are never returned as usable.
2. Arena allocations meet requested alignment and remain within bounds.
3. Zero-size, overflow, invalid alignment and exhaustion are handled explicitly.
4. High-water and remaining-capacity counters are correct.
5. Model bytes can be viewed read-only without copying the full weights.
6. No paging manager, swap, demand paging, paged KV or NUMA policy is introduced.

## WP-04 — CPU identification and safe ISA dispatch

**Goal:** Support AMD and Intel through x86 architectural capabilities.

**Interfaces**
- CpuFeatures: baseline, SSE2, AVX, AVX2, FMA and optional AVX-512 subsets.
- CpuTopology: logical processor/APIC identifiers.
- KernelSet::select(features): selects only valid function pointers.

**File-level changes**
- src/cpu/{mod.rs,cpuid.rs,features.rs,topology.rs}
- src/kernels/{mod.rs,dispatch.rs,scalar.rs}
- tests/cpu/feature_dispatch.rs

**Dependencies:** WP-01.

**Acceptance tests**
1. CPUID is queried with supported leaf bounds.
2. AVX dispatch requires AVX, OSXSAVE and required XCR0 state.
3. Forced feature masks prove unsupported kernels are never called.
4. Scalar implementation remains available on every target.
5. AMD and Intel use capability checks; vendor strings are diagnostic only.

## WP-05 — Scalar tensor and math reference

**Goal:** Establish correctness before optimized kernels.

**Interfaces**
- TensorView: checked dimensions, strides, dtype and slices.
- QuantizationDescriptor: format, block size, scale type and byte length.
- Scalar dot, matvec, matmul, RMSNorm, RoPE, SiLU, softmax and argmax.

**File-level changes**
- src/tensor/{mod.rs,shape.rs,view.rs,dtype.rs}
- src/quant/{mod.rs,descriptor.rs,q8_0.rs}
- src/kernels/scalar/{matvec.rs,matmul.rs,normalization.rs,rope.rs,activation.rs,softmax.rs}
- tests/numerical/reference_ops.rs
- tools/reference-tests/ host-side reference harness.

**Dependencies:** WP-03, WP-04.

**Acceptance tests**
1. Shape/stride/byte-length overflow is rejected.
2. Known-answer tests cover every operation.
3. Differential tests compare against a trusted host FP32 reference.
4. NaN/Inf and edge cases are tested.
5. Scalar backend runs in QEMU and host test mode with identical defined semantics.

## WP-06 — Model packer and validated model source

**Goal:** Convert one small dense checkpoint into a compact, aligned, memory-mappable boot module.

**Interfaces**
- Versioned model header: magic, version, architecture ID, tokenizer ID, tensor count and section offsets.
- Tensor record: name/hash, dimensions, dtype/quantization and byte range.
- ModelSource::section(range): validated immutable byte slice.

**File-level changes**
- tools/model-packer/ (host Rust or Python; host dependencies permitted).
- src/model/{format.rs,header.rs,tensor_table.rs,loader.rs,source.rs}
- tests/model/{format_validation.rs,corrupt_inputs.rs}
- docs/baremetal/MODEL_FORMAT.md

**Dependencies:** WP-03, WP-05.

**Acceptance tests**
1. Exporter round-trips tensor metadata and content hashes.
2. Target rejects unknown versions, bad checksums, out-of-range offsets, overlapping sections, invalid dimensions and unsupported dtypes.
3. Tensor sections satisfy documented alignment.
4. Loader exposes read-only slices over boot module data without duplicating all weights.
5. No host-tool dependency is present in target binary.

## WP-07 — Tokenizer and sampling baseline

**Goal:** Reproduce the selected model tokenizer exactly before generation integration.

**Interfaces**
- Tokenizer::encode(utf8) returns token IDs.
- Tokenizer::decode(ids) returns bounded UTF-8 output.
- Sampler::select(logits, config, rng).

**File-level changes**
- src/tokenizer/{mod.rs,bpe.rs,vocab.rs,merges.rs,unicode.rs}
- src/sampling/{mod.rs,greedy.rs}
- tools/model-packer/ asset conversion.
- tests/tokenizer/golden_cases.rs

**Dependencies:** WP-06.

**Acceptance tests**
1. Golden token IDs match source tokenizer for ASCII, whitespace, punctuation and multibyte Unicode.
2. Invalid UTF-8 handling is explicit and bounded.
3. Greedy argmax matches reference indices including documented tie-breaking.
4. No unbounded per-token heap growth.

## WP-08 — Contiguous KV-cache and sequence state

**Goal:** Provide a single-sequence bounded cache with straightforward ownership.

**Interfaces**
- KvCache::new(config, arena).
- KvCache::append(layer, position, key, value).
- KvCache::view(layer, range).
- SequenceState: current position, context limit and cache ownership.

**File-level changes**
- src/kv/{mod.rs,config.rs,contiguous.rs,sequence.rs}
- tests/kv/{capacity.rs,append_read.rs,bounds.rs}

**Dependencies:** WP-03, WP-05.

**Acceptance tests**
1. Append/read round-trips exact stored values.
2. Position overflow and context exhaustion return errors, never write out of bounds.
3. Required capacity is computed before generation starts.
4. Cache memory is accounted in the fixed KV arena.
5. No paging, eviction, streaming or tiering in this milestone.

## WP-09 — Single-core dense decoder inference

**Goal:** Generate tokens end-to-end for one small dense decoder using scalar operations.

**Interfaces**
- ModelAdapter::load(source, memory) returns Model.
- Model::prefill(tokens, kv).
- Model::decode(next_token, kv) returns logits.
- InferenceSession::generate(prompt, config).

**File-level changes**
- src/model/dense_decoder/{mod.rs,config.rs,forward.rs,attention.rs,mlp.rs}
- src/inference/{mod.rs,session.rs,prefill.rs,decode.rs}
- src/app/serial_chat.rs
- tests/model/golden_generation.rs

**Dependencies:** WP-05 through WP-08.

**Acceptance tests**
1. Prompt completes in QEMU with serial token output.
2. Intermediate logits compare with trusted reference within documented tolerances.
3. Greedy output matches golden token sequences for multiple prompts.
4. KV positions and context bounds are correct across session reset/reuse.
5. Peak memory stays within declared arenas; exhaustion fails cleanly.

## WP-10 — AVX2/FMA kernel optimization

**Goal:** Optimize measured bottlenecks without changing model semantics.

**Interfaces:** Same kernel signatures as scalar backend; runtime dispatch selects scalar or AVX2/FMA.

**File-level changes**
- src/kernels/x86_64/{mod.rs,avx2/mod.rs,avx2/q8_matvec.rs,avx2/matmul.rs}
- tests/kernels/differential.rs
- benches/prefill.rs and benches/decode.rs or equivalent hardware harness.

**Dependencies:** WP-04, WP-05, WP-09.

**Acceptance tests**
1. Every SIMD kernel passes randomized differential tests against scalar reference.
2. Dispatch is tested with AVX2 disabled and enabled.
3. No AVX instruction executes before feature/state validation.
4. Benchmarks record model hash, CPU, ISA, context, prompt, token count and compiler flags.
5. Merge only with repeatable end-to-end gain and no unacceptable numerical/token regressions.

## WP-11 — Multicore inference

**Goal:** Add AMD/Intel multicore execution after single-core correctness and baselines.

**Interfaces**
- CoreControl::start(core_id, entry).
- WorkerPool::dispatch(job), barrier().
- Deterministic reduction contract for partial results.

**File-level changes**
- src/platform/x86_64/{apic.rs,smp.rs}
- boot/x86_64/ap_trampoline.asm
- src/runtime/{workers.rs,barrier.rs,job.rs}
- tests/qemu/smp-smoke.sh and multicore numerical tests.

**Dependencies:** WP-02, WP-04, WP-09.

**Acceptance tests**
1. AP startup succeeds for available QEMU vCPUs and reports online cores.
2. Worker stacks and per-core state are isolated and bounded.
3. Barrier and job completion tests pass under stress.
4. Greedy output is stable across supported worker counts within declared determinism policy.
5. Scaling reports 1, 2, 4, 8 and available core counts; no linear scaling assumption.

## WP-12 — Second boot path and shared handoff

**Goal:** Add direct firmware boot without changing the inference kernel.

**Interfaces:** Firmware adapter populates the same versioned BootInfo. Shared kernel entry and inference interfaces remain unchanged.

**File-level changes**
- boot/x86_64/uefi/ or selected direct firmware entry.
- src/boot/uefi.rs
- scripts/build-uefi-image.sh
- tests/qemu/uefi-smoke.sh
- docs/baremetal/BOOT_PATHS.md

**Dependencies:** WP-01 through WP-09. Implement after WP-09 so the second path proves a working inference system.

**Acceptance tests**
1. Firmware path enters shared kernel entry with valid normalized BootInfo.
2. Same model generates same golden tokens through both boot paths.
3. No boot-protocol-specific types leak into inference crates.
4. Firmware services are exited or no longer used before normal inference begins, except documented boot-time facilities.

## First release definition of done

Complete WP-01 through WP-09. The standard-loader QEMU image boots, accepts a prompt over serial and generates tokens from one small dense model using scalar Rust kernels and bounded contiguous memory. WP-10 through WP-12 are subsequent optimization/platform milestones. Target runtime has no OS, libc, GPU, driver or vendor compute SDK dependency.

## WP-00 — Reproducible Ubuntu/KVM development environment

**Goal:** Provide an isolated development and test environment on the existing Ubuntu 26.04 host without changing its OS, bootloader or disk layout.

**Interfaces / artifacts**
- Documented KVM/libvirt VM definition for `BareMetalInfer-Dev`.
- Host script or documented command to launch freestanding QEMU/KVM images.
- Machine-readable benchmark environment metadata.

**File-level changes**
- docs/baremetal/DEVELOPMENT_VM_SETUP.md
- scripts/vm/ (optional provisioning and launch scripts; add only after commands are validated)
- tests/qemu/ (host-launched freestanding smoke test integration)
- benchmarks/environment/ (capture host, guest, QEMU and CPU-feature metadata)

**Dependencies:** Existing Ubuntu 26.04 host with AMD SVM enabled in firmware.

**Acceptance tests**
1. KVM is available and a clean Ubuntu guest boots.
2. Guest CPU feature visibility and vCPU topology are recorded.
3. Development guest builds the hosted adapter and host-side tests.
4. Host-launched QEMU/KVM boots the freestanding test image with serial output.
5. No nested virtualization is required for the initial workflow.
6. Host OS, boot configuration and physical disk layout remain unchanged.
7. Benchmark artifacts identify Linux VM versus freestanding QEMU/KVM execution; neither is mislabeled as physical bare metal.

