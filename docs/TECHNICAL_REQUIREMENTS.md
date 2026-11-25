# Infer — Technical Requirements

**Status:** Initial technical baseline | **Version:** 0.1 | **Date:** 2026-11-26  
**Parent:** [Product Requirements](PRODUCT_REQUIREMENTS.md)

## 1. Purpose and scope

This document turns the product baseline into testable engineering constraints. Initial implementation: Rust, native Linux, one AMD RX 9070, selected ggml/llama.cpp compute infrastructure, HIP/ROCm, one host and one loaded model. Build the generic foundation before the Qwen3.8-Flash-Next adapter.

This is a requirements baseline, not a claim that the requirements are already implemented or validated.

## 2. Fixed decisions

| Decision | Baseline |
|---|---|
| Implementation | Rust stable; pin MSRV with initial dependency set |
| Build | Cargo; CMake/Ninja for native ggml/HIP components |
| Device | RX 9070 / gfx1201 / HIP |
| OS | Native Ubuntu Linux |
| Compute reuse | Selected ggml/llama.cpp backend, graph and kernel infrastructure via narrow FFI |
| Infer ownership | Request scheduling, adapters, global budgets, residency policy, planning, config and telemetry |
| Topology | Single process, host, GPU and loaded model |
| First model | Exact Qwen3.8-Flash-Next GSQ-RCO artifact after metadata validation |
| Memory | Explicit tiered residency, not transparent GPU paging |
| Delivery | Generic foundation, then Qwen adapter |

## 3. Functional requirements

### 3.1 Runtime and device discovery

- TR-001: Enumerate backend devices with name, architecture, total/free memory and capabilities.
- TR-002: Validate selected HIP device against compiled gfx target before allocation.
- TR-003: Record runtime, driver, HIP/ROCm, ggml revision, Infer revision and build options.
- TR-004: Support explicit device selection; never silently choose the integrated GPU when the requested device is unavailable.
- TR-005: Fail early when an operation, tensor type or quantization is unsupported.

### 3.2 Model loading and metadata

- TR-010: Separate file parsing, tensor storage/quant decoding, architecture metadata and execution topology.
- TR-011: Validate model identity, architecture, tensor names/shapes/types, tokenizer assets, shards and declared context.
- TR-012: Never infer architecture from filename alone.
- TR-013: Select adapters through validated metadata and capability matching.
- TR-014: Support mmap-backed model sources when format and backend permit.
- TR-015: Reject unsupported custom tensor types precisely; never reinterpret custom GSQ-RCO IDs as stock ggml quant types.

### 3.3 Execution and generation

- TR-020: Represent model execution as a logical graph/plan independent of physical residency.
- TR-021: Treat prefill and decode as distinct workload modes and metrics.
- TR-022: Support incremental output, cancellation, bounded generation and deterministic cleanup.
- TR-023: Separate tokenizer, chat template, sampler and stopping criteria from model topology.
- TR-024: Keep hardware/backend decisions out of model adapters.
- TR-025: Resolve provider/kernel selection at plan or dispatch boundaries, not per tensor element.

### 3.4 Tiered memory and virtual residency

- TR-030: Define explicit tiers: device VRAM, pinned host RAM, pageable host RAM and file/mmap backing.
- TR-031: Every managed resource has stable identity, class, size, alignment, ownership, residency, mutability/version, last-use fence and optional backing source.
- TR-032: One global coordinator accounts for weights, experts, PLE/lookups, attention KV, recurrent state, workspaces, staging and safety reserve.
- TR-033: Reserve budget before provider allocation; roll back reservation on failure.
- TR-034: Model residency changes as explicit state transitions.
- TR-035: Async transfers return completion tickets/fences; source and destination remain owned until dependencies complete.
- TR-036: Forbid eviction while leases, pending copies or GPU uses are active.
- TR-037: Bound pinned-host allocations; fallback or fail according to explicit policy.
- TR-038: Never assume pageable-host transfers are asynchronous or equivalent to pinned memory.
- TR-039: No NVMe/file-backed reads in the ordinary token-critical path by default.
- TR-040: Prevent thrashing with resource-specific policies and transfer budgets; expose hits, misses, bytes, waits and evictions.
- TR-041: Keep configurable VRAM safety reserve for runtime, workspace, display and transient allocations.
- TR-042: Monitor host MemAvailable and pin limits; do not lock the full model by default.

#### Resource-specific requirements

| Resource | Required policy |
|---|---|
| Attention KV | Paged/block residency; sequence-aware lifetime; host-backed cold blocks; device residency before attention consumes data |
| Recurrent/SSM state | Dedicated manager and state lifecycle; not ordinary attention KV |
| MoE experts | Expert-aware cache; profile/frequency/LRU baseline; prefetch only when measured useful |
| PLE/lookup | Separate I/O/cache policy; mmap or bounded host cache; selective staging |
| Dense/layer weights | Prefer persistent VRAM residency; optional staged execution only after measured correctness/throughput |
| Workspace | Device-local, pre-reserved and non-evictable while graph work is active |
| Transfer staging | Bounded pools with explicit ownership and completion |

### 3.5 KV and recurrent-state semantics

- TR-050: Generic memory must not assume all models have attention KV.
- TR-051: KV geometry (tokens/page, layers, heads, type and alignment) is supplied by adapter capabilities.
- TR-052: KV pages are versioned by sequence and position; append, truncate, fork/share and release are explicit.
- TR-053: Host-resident KV must become device-accessible before the consuming attention operation.
- TR-054: Recurrent state has independent create/update/snapshot/restore/release operations.
- TR-055: Prefix reuse/checkpoint restore validates model identity, adapter version, execution config and state format.
- TR-056: KV quantization is a capability, not a global assumption; test numerical quality for every supported mode.

### 3.6 Scheduling and concurrency

- TR-060: Infer owns request admission, cancellation, fairness and high-level scheduling.
- TR-061: Initial release may serialize model execution while retaining asynchronous transfer pipelines.
- TR-062: Scheduler coordinates execution dependencies, reservations, transfer tickets and resource leases.
- TR-063: Never evict or overwrite before the last GPU completion event.
- TR-064: Bound queued requests, pending transfers and pinned staging buffers.
- TR-065: Cancellation/error must safely drain or detach in-flight work before releasing resources.
- TR-066: Distinguish ggml backend graph/device scheduling from Infer request scheduling and residency planning.

### 3.7 Interfaces and operations

- TR-070: CLI supports device inspection, model validation, config validation, inference and benchmarks.
- TR-071: Structured logs use stable error codes and request/correlation IDs.
- TR-072: Telemetry covers prefill/decode, TTFT, inter-token latency, memory by tier/class, transfers, residency, cache behavior and failures.
- TR-073: OpenAI-compatible HTTP API is P1; core interfaces must permit it without model-core changes.
- TR-074: Version and validate configuration; report effective values and unsupported/ignored options.
- TR-075: No network dependency for local inference after installation and asset acquisition.

## 4. Non-functional requirements

### Correctness and safety
- NFR-001: No use-after-free, double-free, stale device pointer or early eviction under asynchronous execution.
- NFR-002: All memory transitions are validated and observable.
- NFR-003: On transfer failure, destination is not marked resident; source remains valid until cleanup/recovery.
- NFR-004: Compare outputs against a trusted reference for supported models and residency modes.
- NFR-005: Errors cannot leave budget accounting, leases or provider allocations inconsistent.

### Performance
- NFR-010: Avoid device-wide synchronization in steady-state decode; use stream/event dependencies.
- NFR-011: Overlap transfer and compute only when dependencies, pinned memory and hardware allow; measure actual overlap.
- NFR-012: No mandatory host round-trip per token when data is already resident.
- NFR-013: Track transfer amplification and thrashing; optimize prefill and decode separately.
- NFR-014: No performance target without a frozen baseline and matched workload.

### Extensibility and maintainability
- NFR-020: Core crates do not import HIP, CUDA or model-family implementation modules.
- NFR-021: Unsafe Rust and native FFI are isolated in a small interoperability crate with documented invariants.
- NFR-022: Introduce interfaces at real change boundaries; avoid one-trait-per-type abstractions.
- NFR-023: Apply dependency inversion at subsystem boundaries, not inside tight tensor loops.
- NFR-024: One module owns each resource lifecycle; shutdown order is explicit.
- NFR-025: No global mutable state; runtime/device/context ownership is explicit.
- NFR-026: Modular monolith initially; no microservices or dynamic plugin loader in P0.

## 5. Rust/native interoperability constraints

- Engine, application and policy code are Rust.
- ggml/llama.cpp and HIP kernels remain native components where required.
- Provide a small C ABI shim and Rust FFI crate with opaque handles; do not expose C++ classes, STL containers or exceptions across ABI.
- Native exceptions and Rust panics must not unwind across the C ABI.
- Every opaque handle has one documented owner and matching destroy operation.
- Completion tickets retain buffer ownership until GPU completion.
- Pin ggml/llama.cpp to a known commit and record patches; avoid unstable internal headers unless wrapped and tested.
- Keep proven quantized execution in ggml initially; add custom kernels only after profiling identifies material value.

## 6. Initial Rust workspace boundaries

Suggested modular monolith:
- infer-cli: CLI and configuration.
- infer-core: model-neutral request, generation and scheduling types.
- infer-model: registry and adapter contracts.
- infer-memory: coordinator, registry, residency state machine and policies.
- infer-runtime: execution plans, dispatch and transfer dependencies.
- infer-ggml-sys: raw FFI and native build integration.
- infer-ggml: safe wrappers for ggml contexts, buffers, graphs and events.
- infer-observability: metrics, structured logs and tracing.
- infer-testkit: fixtures, fake providers and reference comparisons.
- infer-model-qwen38: first model adapter, added after generic foundation.
- native/ggml-bridge: small C ABI shim built with CMake/Ninja.

Do not create a crate for every folder. Split only where ownership, portability or independent testing benefit.

## 7. Verification and acceptance tests

| Test | Verification | Pass condition |
|---|---|---|
| VT-001 | Device discovery | RX 9070/gfx1201 correctly identified and selected |
| VT-002 | FFI lifecycle | Repeated create/use/destroy without leaks |
| VT-003 | Memory budget | Concurrent reservations cannot exceed configured budget |
| VT-004 | Residency | Host-to-VRAM transition preserves bytes and metadata |
| VT-005 | Lease safety | Eviction waits/rejects while lease or GPU event is outstanding |
| VT-006 | Transfer failure | Source remains valid and accounting consistent |
| VT-007 | KV paging | Append/read/evict/restore matches reference attention output |
| VT-008 | Recurrent state | Snapshot/restore independent of KV lifecycle |
| VT-009 | Backpressure | Pending transfer and staging memory remain bounded |
| VT-010 | Generic adapter | Test adapter runs without Qwen-specific core dependency |
| VT-011 | Qwen metadata | Exact architecture, tensor map, custom quant IDs and context validated |
| VT-012 | Qwen execution | First adapter matches reference within defined tolerance |
| VT-013 | Long context | Planted-depth and broader quality checks pass at tested context |
| VT-014 | Benchmark | At least three matched repeats with median/range and memory minima |
| VT-015 | Shutdown | No active leases or leaked native allocations after cancel/error/shutdown |

## 8. Benchmark protocol

Record exact model/artifact and hashes; architecture and quantization; Infer and ggml commits; compiler/build; ROCm/HIP/driver; OS/kernel/device; context, KV type/page size/resident budget, expert/PLE policy and reserve; exact prompt/output tokens; warm/cold state and prefix reuse; prefill/decode, TTFT, inter-token latency, peak/lowest-free VRAM, host MemAvailable, pinned allocations, transfer bytes/latency, cache hits, correctness and warnings.

Run at least three matched repetitions and report median/range. Separate cold load, warm process, fresh prompt, reused prefix, prefill and decode. Dashboard observations or incomparable runs are not benchmarks.

## 9. Evidence constraints from Qwen3.8 investigation

The first candidate is specifically **Qwen3.8-Flash-Next GSQ-RCO IQ1_M**, not every Qwen3.8-family model.

Recorded artifact metadata:
- GGUF architecture label: qwen4exp;
- 48 blocks, embedding 2,560;
- 24 attention heads / 2 KV heads;
- full attention interval 4 with hybrid SSM/recurrent blocks;
- 256 experts / 10 active experts; expert FFN 640;
- declared context 262,144;
- separate PLE shard about 26.82 GiB; combined files about 54.4 GiB;
- mixed GSQ-RCO quantization; filename IQ1_M does not mean all tensors use IQ1_M.

One recorded PolyStrata runtime configuration:
- 131,072 context, q4_0 KV, 32,768 resident KV cells;
- 0.84 GiB pinned host KV at startup;
- about 23.42 GiB expert arena, about 9.90 GiB expert cache in VRAM, 819 MiB MTP draft layer and about 386 MiB free VRAM in a telemetry snapshot.

Evidence constraints:
- These are artifact-specific metadata and one engine/build/config observation, not generic requirements or an Infer memory plan.
- 262,144 is native context metadata for this exact artifact; 512K is experimental rope scaling and was not validated in the cited investigation.
- Existing request timings were not controlled long-context benchmarks; do not use them as performance targets.
- Verify exact custom tensor type support before choosing adapter/decoder.
- Tiel Coder results/settings are a separate model/build and must not be transferred to Qwen3.8.

## 10. Open technical questions

1. Which pinned ggml revision and build options provide the required HIP/gfx1201 operations?
2. Is the ggml HIP backend sufficient for copies/events/pinned buffers, or is a supplemental HIP shim required?
3. Which Qwen3.8 GSQ-RCO tensor types are upstream-supported and which need an isolated extension?
4. What resource granularity avoids excessive expert/PLE transfer amplification?
5. Which KV page sizes (e.g. 128/256 tokens) perform best for target shapes and PCIe?
6. Which host allocation strategy is reliable under target ROCm/kernel/pinning limits?
7. What numerical tolerances are acceptable for quantized/streamed execution?
8. Which exact model/tokenizer/template artifacts are authoritative?

Resolve through source inspection and small experiments, not assumptions.