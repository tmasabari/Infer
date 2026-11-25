# Infer

A modular, local-first inference engine for constrained hardware.

## Project baseline

The initial product targets one AMD Radeon RX 9070 (16 GB VRAM) on native Linux with ROCm/HIP. The engine is implemented in Rust and reuses selected ggml/llama.cpp compute infrastructure through a narrow native interoperability boundary.

The first delivery is a generic inference foundation. Qwen3.8-Flash-Next is the first model adapter after the core contracts and memory lifecycle are validated.

## Context documents

- [Product requirements](docs/PRODUCT_REQUIREMENTS.md) — user value, Pareto priorities, scope and success criteria.
- [Technical requirements](docs/TECHNICAL_REQUIREMENTS.md) — functional/non-functional requirements, Rust/FFI constraints and verification.
- [Reference architecture](docs/REFERENCE_ARCHITECTURE.md) — subsystem boundaries, model adapters, tiered memory and execution flow.

## Core architectural direction

- Infer owns model-neutral request scheduling, adapters, execution/residency planning, global memory budgets, telemetry and product lifecycle.
- ggml/llama.cpp supplies selected tensor, graph, quantized-kernel and backend capabilities.
- HIP/ROCm is the first compute path.
- VRAM, pinned host RAM, pageable host RAM and mmap/file backing are explicit memory tiers.
- Attention KV, recurrent state, experts, PLE and workspaces have different lifecycle and residency policies.
- The first release is a modular monolith: one process, one host, one GPU and one loaded model.

## Current status

Documentation baseline only. No inference engine implementation or performance claim is implied by these documents. See the technical requirements for unresolved source/API and model-compatibility questions.

## Development principle

Apply the Pareto principle in every phase: build the smallest set of capabilities that creates the greatest user value. Preserve correctness and observability before optimizing.