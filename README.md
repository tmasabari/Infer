# Infer

A modular, local-first inference engine for constrained hardware.

## Project baseline

The initial product targets one AMD Radeon RX 9070 (16 GB VRAM) on native Linux with ROCm/HIP. The engine is implemented in Rust and reuses selected ggml/llama.cpp compute infrastructure through a narrow native interoperability boundary.

The first delivery is a generic inference foundation. Qwen3.8-Flash-Next is the first model adapter after the core contracts and memory lifecycle are validated.

## Context and design documents

### BareMetalInfer workstream (`BareMetalInfer` branch)

This branch also contains a separate CPU-only, x86-64 bare-metal inference workstream. It is distinct from the RX 9070/ROCm product baseline described above; do not mix their runtime, hardware or acceptance criteria. The BareMetalInfer architecture and implementation context is maintained under [`docs/baremetal/`](docs/baremetal/):

- [Architecture comparison and decision](docs/baremetal/ARCHITECTURE_COMPARISON_AND_DECISION.md) — comparison of `alesha-pro/llm.asm` and `Danmoreng/qwen35-cpu`, selected hybrid reuse strategy and MVP model direction.
- [BareMetalInfer architecture](docs/baremetal/BAREMETAL_INFER_ARCHITECTURE.md) — platform boundaries, memory model and runtime invariants.
- [Platform and language strategy](docs/baremetal/PLATFORM_AND_LANGUAGE_STRATEGY.md) — Rust `no_std`, assembly boundary, boot and ISA policy.
- [Development VM setup](docs/baremetal/DEVELOPMENT_VM_SETUP.md) — Ubuntu/KVM/QEMU development and validation environment.
- [Implementation roadmap](docs/baremetal/IMPLEMENTATION_ROADMAP.md) and [work packages](docs/baremetal/IMPLEMENTATION_WORK_PACKAGES.md) — delivery sequence and acceptance gates.



- [Product requirements](docs/PRODUCT_REQUIREMENTS.md) — user value, Pareto priorities, scope and success criteria.
- [Technical requirements](docs/TECHNICAL_REQUIREMENTS.md) — functional/non-functional requirements, Rust/FFI constraints and verification.
- [Reference architecture](docs/REFERENCE_ARCHITECTURE.md) — subsystem boundaries, model adapters, tiered memory and execution flow.
- [Phase 1 MVP design](docs/PHASE_1_MVP_DESIGN.md) — implementation-level design, contracts, native boundary, execution flow, testing, work packages and exit criteria.
- [Phase 1 component designs](docs/PHASE_1_COMPONENT_DESIGNS.md) — detailed designs for every Phase 1 component, with comic-style walkthroughs, activity diagrams, flowcharts, lifecycle rules and verification criteria.

## Core architectural direction

- Infer owns model-neutral request scheduling, adapters, execution/residency planning, global memory budgets, telemetry and product lifecycle.
- ggml/llama.cpp supplies selected tensor, graph, quantized-kernel and backend capabilities.
- HIP/ROCm is the first compute path.
- VRAM, pinned host RAM, pageable host RAM and mmap/file backing are explicit memory tiers.
- Attention KV, recurrent state, experts, PLE and workspaces have different lifecycle and residency policies.
- The first release is a modular monolith: one process, one host, one GPU and one loaded model.

## Current status

Documentation baseline only. No inference engine implementation or performance claim is implied by these documents. See the technical requirements and Phase 1 design documents for unresolved source/API and model-compatibility questions.

## Development principle

Apply the Pareto principle in every phase: build the smallest set of capabilities that creates the greatest value for users. Preserve correctness and observability before optimizing.
