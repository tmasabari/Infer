# Infer — Product Requirements

**Status:** Initial product baseline | **Version:** 0.1 | **Date:** 2026-11-26

## 1. Product vision

Infer is a local-first, modular inference engine for running capable language models on constrained consumer hardware. It coordinates model execution and multiple memory tiers so limited GPU VRAM can be used efficiently without hiding the cost of moving data to and from host memory.

The first target is one AMD Radeon RX 9070 (16 GB VRAM), one host and one loaded model. The first milestone is a generic engine foundation; Qwen3.8-Flash-Next is the first demanding model adapter, not the definition of the product.

**Product promise:** make more useful local inference possible on limited hardware through predictable memory management, efficient execution and replaceable model/backend components.

## 2. Target users and jobs

Primary users are local-AI developers and technically capable enthusiasts who:
- have a consumer GPU with limited VRAM and substantial system RAM;
- want to run models or contexts that exceed the GPU's resident-memory budget;
- value local execution, open-source components and reproducible measurements;
- need long-context and MoE workloads without a cloud inference dependency;
- want to add model families without rewriting the engine.

Core jobs:
1. Run supported models locally with minimal manual tensor/cache placement.
2. Extend GPU capacity using host memory without uncontrolled swapping.
3. Maintain interactive generation while KV, recurrent state and expert working sets grow.
4. Diagnose performance and memory bottlenecks through actionable telemetry.
5. Add model architectures and memory policies without rewriting unrelated subsystems.

## 3. Product principles

1. **Local-first:** no remote inference or telemetry by default.
2. **Performance with evidence:** optimize end-to-end workloads, not isolated kernel numbers.
3. **Memory is a product capability:** VRAM, pinned RAM, pageable RAM and storage are distinct tiers.
4. **Predictability over magic:** expose residency, budgets and transfer costs; prevent pathological paging.
5. **Model independence:** model-specific behavior belongs behind adapters and declared capabilities.
6. **Backend independence:** product/core logic must not depend directly on HIP, CUDA or platform APIs.
7. **Simple first release:** one process, one host, one GPU, one loaded model.
8. **Reproducibility:** benchmarks identify model, quantization, build, hardware, workload, cache state and memory minima.

## 4. Pareto roadmap: highest-value 20%

Every phase applies the 80/20 rule: implement the smallest set of capabilities that creates the greatest user value. Do not add technically interesting features without a demonstrated user benefit.

### P0 — first useful release

| ID | Product requirement | User value | Acceptance outcome |
|---|---|---|---|
| PR-001 | Execute a supported model on RX 9070 through HIP | Proves the target deployment | Repeatable load, prompt, decode and unload test |
| PR-002 | Enforce one global memory budget | Prevents independent caches exhausting VRAM | Allocations reserve against one budget and respect safety reserve |
| PR-003 | Explicit tiered residency for VRAM and host memory | Uses host RAM as a deliberate capacity tier | Safe host-to-device resource transition with correctness checks |
| PR-004 | Paged/dynamic attention KV residency | Supports context growth beyond resident KV capacity | Long request completes without corruption, deadlock or unbounded memory |
| PR-005 | Generic model-adapter boundary | Allows future architectures | First adapter adds no model-specific branches to core |
| PR-006 | Reliable streaming generation and cancellation | Makes engine usable interactively | Incremental tokens, cancellation, bounded output and clean termination |
| PR-007 | Memory and performance telemetry | Makes bottlenecks diagnosable | Reports prefill/decode, memory tiers, transfers, residency and cache metrics |
| PR-008 | Reproducible CLI and versioned configuration | Makes runs repeatable | Effective configuration and runtime identity are recorded |

### P1 — next high-value capabilities

| ID | Product requirement | User value | Acceptance outcome |
|---|---|---|---|
| PR-101 | Expert-aware cache and prefetch | Reduces MoE transfer stalls | Hit rate, transfer bytes and prefetch utility measured |
| PR-102 | mmap-backed cold model/lookup data | Avoids eagerly copying all data into RAM | Bounded host cache; storage reads observable |
| PR-103 | OpenAI-compatible local streaming API | Integrates existing clients | Streaming chat completions and health/model endpoints |
| PR-104 | Multiple residency policies | Enables workload-specific tuning | Fixed-resident and dynamic/paged policies comparable |
| PR-105 | Benchmark harness and correctness checks | Prevents speed-only regressions | Matched workloads run with reference comparisons |
| PR-106 | Second model adapter | Proves extensibility | New architecture added without core redesign |

### P2 — defer until measured value justifies the cost

- Multi-GPU, tensor/expert parallelism and distributed inference.
- CUDA and Vulkan providers.
- Windows-native runtime support; initial validation is native Linux.
- Concurrent multi-model serving and multi-tenant scheduling.
- Dynamic third-party plugin loading.
- Training and fine-tuning.
- Agent orchestration and application-specific workflows.
- Speculative decoding/MTP until baseline correctness and performance are stable.
- Transparent OS-like GPU page-fault emulation.

## 5. Initial target environment

| Dimension | Initial target |
|---|---|
| GPU | AMD Radeon RX 9070, 16 GB class, gfx1201 |
| CPU | AMD Ryzen 7 9700X |
| RAM | 64 GB installed; available memory varies by workload |
| OS | Native Ubuntu Linux with ROCm/HIP |
| Language | Rust for engine/application code |
| Compute foundation | Selected ggml/llama.cpp infrastructure and kernels |
| Deployment | Single local process |
| First model | Exact Qwen3.8-Flash-Next GSQ-RCO artifact, validated at load time |

This is a validation environment, not a guarantee that every model or quantization fits or performs well.

## 6. User experience requirements

### Setup
- Inspect devices and effective backend before loading.
- Configure model path, context, memory budgets, KV policy and logging.
- Unsupported formats, tensor types and architecture features fail with actionable diagnostics.
- Never silently substitute an unsupported quantization or claim an experimental feature is validated.

### Inference
- Support prompt ingestion, incremental token output, cancellation and bounded output.
- Distinguish prefill from decode.
- Identify fresh prompts, prefix reuse and checkpoint restore.
- Account for template/special tokens, recurrent state and output reservation in context admission.

### Memory
- Show total/free VRAM, engine budget, workspace reserve, host/pinned memory and storage cache.
- Report residency and transfer bytes/latency.
- Prevent eviction while a GPU operation still owns a resource.
- Expose policy settings and trade-offs.

## 7. Product quality attributes

| Attribute | Expectation |
|---|---|
| Correctness | Logical model behavior is preserved across residency tiers |
| Performance | No avoidable synchronous transfer in the steady-state token path |
| Stability | Bounded memory; no use-after-free or transfer-related deadlock |
| Extensibility | New adapters/providers do not require unrelated core changes |
| Observability | Resource movement and budget decisions are diagnosable |
| Privacy | No network access required after local assets are installed |
| Operability | Validated configuration, structured errors and clean shutdown |

## 8. Non-goals and guardrails

- Infer is not initially a wholesale llama.cpp fork or replacement.
- Infer does not initially implement a new quantized kernel library.
- Host RAM is not equivalent to VRAM; PCIe and copy costs remain visible.
- Arbitrary 262K/512K context is not promised for every model.
- Architecture and context metadata must come from the exact model artifact, not a family name.
- Qwen3.8-Flash-Next observations must not be generalized to other Qwen variants.
- No benchmark result is accepted without model/build identity, workload and evidence classification.

## 9. Success metrics

The first product milestone succeeds when Infer can:
1. load and execute a supported generic model on RX 9070;
2. operate under a configured VRAM budget while staging eligible resources through host memory;
3. run a paged-KV workload beyond the configured resident KV budget;
4. preserve output correctness across resident and streamed execution;
5. explain memory pressure and performance through telemetry;
6. add Qwen3.8 through an adapter without model-specific logic leaking into the core.

Numeric performance targets are deferred until a reproducible generic baseline exists.