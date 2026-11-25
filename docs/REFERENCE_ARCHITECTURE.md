# Infer — Reference Architecture

**Status:** Initial architecture baseline | **Version:** 0.1 | **Date:** 2026-11-26  
**Style:** Modular monolith, ports-and-adapters, explicit resource ownership  
**Language:** Rust | **Initial compute:** ggml/llama.cpp + HIP/ROCm on RX 9070

## 1. Architecture intent

Infer owns the inference product architecture while reusing proven compute and tensor infrastructure. It is not a wholesale llama.cpp fork and does not initially reimplement quantized kernels.

Separate four concerns:
1. **Model semantics:** adapter describes logical architecture and model state.
2. **Execution planning:** Infer schedules requests and decides when resources must be resident.
3. **Memory policy:** Infer owns budgets, residency, prefetch and eviction.
4. **Compute execution:** ggml/llama.cpp backends execute supported graphs/kernels; HIP supplies the AMD path.

A narrow C ABI boundary isolates native code and unsafe Rust.

## 2. Context and container diagram

~~~mermaid
flowchart TB
    Client[CLI / future HTTP clients] --> App[Infer Application]
    App --> Core[Inference Core]
    Core --> Scheduler[Request Scheduler]
    Core --> Adapter[Model Adapter]
    Adapter --> Plan[Logical Execution Plan]
    Scheduler --> Planner[Execution and Residency Planner]
    Plan --> Planner
    Planner --> Memory[Memory Coordinator]
    Planner --> Runtime[Compute Runtime]
    Memory --> Providers[Memory Providers]
    Runtime --> GGML[Safe ggml Adapter]
    GGML --> FFI[C ABI / FFI Boundary]
    FFI --> HIP[ggml HIP Backend]
    HIP --> GPU[RX 9070]
    Providers --> Host[Host RAM / pinned RAM / mmap]
    Providers --> GPU
    Core --> Obs[Telemetry / Tracing]
    Memory --> Obs
    Runtime --> Obs
~~~

Application and core are model-neutral. Qwen is a compile-time/module adapter in the first release, not a dynamically loaded third-party binary.

## 3. Components and ownership

| Component | Owns | Must not own |
|---|---|---|
| Application / CLI | Config, commands, process lifecycle, user errors | Model math, HIP handles |
| Inference Core | Request lifecycle, generation, cancellation, sessions | Device allocation |
| Request Scheduler | Admission, ordering, backpressure, execution windows | Tensor kernels |
| Model Registry | Adapter discovery and capability matching | Backend selection, memory policy |
| Model Adapter | Architecture interpretation, logical graph, state semantics | HIP calls, VRAM budget |
| Execution Planner | Backend-compatible work and resource requirements | Model file parsing |
| Memory Coordinator | Global budgets, reservations and accounting | Model-specific tensor math |
| Residency Manager | Resource location/state and leases | GPU kernel implementation |
| Transfer Scheduler | Copy dependencies, prefetch and completion tickets | Eviction policy decisions |
| Eviction Policy | Candidate ranking per resource class | Raw HIP calls |
| Memory Provider | Allocate/free/read/write in one physical tier | Global policy |
| Compute Runtime | Submit graph work and expose completion fences | Model-specific cache heuristics |
| ggml Adapter | Safe wrappers for contexts, graphs, buffers and events | Request policy |
| Native FFI shim | C ABI and native handle lifecycle | Product configuration |
| Observability | Metrics, events and traces | Runtime policy mutation |

One module owns each allocation and resource lifecycle. Ownership transfer is explicit; raw pointers do not escape safe wrappers.

## 4. Dependency direction

~~~mermaid
flowchart LR
    App[Application] --> Core[Core]
    Core --> Ports[Core Contracts]
    Model[Model Adapter] --> Ports
    Mem[Memory Subsystem] --> Ports
    Runtime[Compute Runtime] --> Ports
    GGML[ggml Interop Adapter] --> Runtime
    FFI[Native C ABI] --> GGML
    HIP[HIP / ggml backend] --> FFI
~~~

Policy modules depend on abstractions they own; infrastructure implements them. Core crates do not import HIP, rocBLAS, ggml internal headers or model-family implementation modules.

Avoid abstractions that do not protect a real change boundary. In hot tensor execution, resolve backend/kernel choices at plan or dispatch boundaries rather than virtual-dispatching every operation.

## 5. Model adapter architecture

An adapter describes model semantics. It does not allocate VRAM or select HIP devices.

Conceptual Rust contract:

~~~rust
pub trait ModelAdapter: Send + Sync {
    fn id(&self) -> ModelArchitectureId;
    fn probe(&self, metadata: &ModelMetadata) -> ProbeResult;
    fn capabilities(&self) -> ModelCapabilities;
    fn build_plan(
        &self,
        metadata: &ModelMetadata,
        request: &ModelRequest,
    ) -> Result<LogicalExecutionPlan>;
    fn state_layout(&self) -> ModelStateLayout;
}
~~~

Capabilities may describe dense/MoE/hybrid blocks, attention KV geometry, recurrent-state semantics, expert topology, PLE/auxiliary lookups, tokenizer/template requirements, context and positional encoding.

Build a generic test adapter first. Add Qwen3.8 only after the contracts are tested. The recorded Qwen3.8-Flash-Next artifact uses qwen4exp metadata, hybrid attention/SSM blocks, MoE expert tensors and a separate PLE shard. Do not assume every Qwen3.8 model has this topology.

## 6. Tiered virtual memory

### 6.1 Logical resource model

Infer exposes a logical resource handle, not a permanently stable device pointer.

~~~rust
pub struct ResourceDescriptor {
    pub id: ResourceId,
    pub class: ResourceClass,
    pub size_bytes: u64,
    pub alignment: u64,
    pub mutability: Mutability,
    pub source: Option<BackingStoreId>,
}

pub enum MemoryTier {
    DeviceVram,
    PinnedHost,
    PageableHost,
    FileMapped,
}

pub enum ResidencyState {
    Backed,
    HostResident,
    TransferToDevice,
    DeviceResident,
    TransferToHost,
    EvictPending,
    Failed,
}
~~~

A device lease is granted only after the resource is device-resident and transfer dependencies have completed.

### 6.2 Physical tiers

| Tier | Purpose | Guardrail |
|---|---|---|
| Device VRAM | Active compute, hot KV, experts and workspace | Strict global budget and reserve |
| Pinned host RAM | Transfer staging and warm resources | Bounded pool; pin only when needed |
| Pageable host RAM | Host-resident resources and caches | Monitor OS available memory; async not assumed |
| File/mmap | Cold model and PLE backing | Not in token-critical path by default |

This is not transparent GPU virtual memory. Kernels still need device-accessible addresses. Infer explicitly plans and completes residency transitions before submission.

### 6.3 Memory subsystem

~~~mermaid
flowchart TB
    C[Global Memory Coordinator]
    C --> R[Resource Registry]
    C --> M[Residency Manager]
    C --> T[Transfer Scheduler]
    C --> E[Resource-specific Eviction Policies]
    C --> O[Memory Telemetry]
    M --> V[HIP / ggml VRAM Provider]
    M --> P[Pinned Host Provider]
    M --> H[Pageable Host Provider]
    M --> F[mmap Backing Provider]
    T --> X[Events and Completion Tickets]
~~~

### 6.4 Global budget

~~~text
VRAM budget =
    dense/model weights
  + expert cache
  + resident attention KV pages
  + recurrent state
  + PLE/lookup staging
  + compute workspace
  + transfer staging
  + runtime/display safety reserve
~~~

Every allocation reserves bytes with the coordinator before provider allocation. On success, commit the reservation; on failure, roll it back. Release budget only after deallocation and all device uses complete.

### 6.5 Resource-specific policy

| Resource | Residency design |
|---|---|
| Attention KV | Fixed token pages; sequence-aware lifetime; hot pages on GPU, cold pages in host; prefetch for attention requirements |
| Recurrent/SSM state | Separate persistent state objects and checkpoint semantics |
| Experts | Expert-aware cache, optional profile/frequency ranking and next-layer prefetch |
| PLE | mmap source and bounded host cache; selective staging; no default full pinned lock |
| Dense/layer weights | Prefer VRAM residency; optional layer/tensor staging after safe overlap is measured |
| Workspace | Reserved device allocation; tied to execution fence; never evict while active |

## 7. Residency state machine and leases

~~~mermaid
stateDiagram-v2
    [*] --> Backed
    Backed --> HostResident: load / page in
    HostResident --> TransferToDevice: reserve VRAM + copy
    TransferToDevice --> DeviceResident: copy completion
    DeviceResident --> TransferToHost: leases drained; eviction
    TransferToHost --> HostResident: copy completion
    DeviceResident --> EvictPending: no lease; last-use fence pending
    EvictPending --> HostResident: fence complete; release VRAM
    HostResident --> Backed: discard clean reloadable copy
    TransferToDevice --> Failed: transfer error
    TransferToHost --> Failed: transfer error
    Failed --> HostResident: recover from valid source
~~~

Safety invariants:
- No eviction while a device lease exists.
- No device lease before transfer completion.
- Transfer tickets retain source and destination ownership until completion.
- Dirty data is not discarded without updating a valid backing copy.
- Errors preserve a valid copy or fail the request without corrupting accounting.
- Cancellation never frees memory referenced by in-flight GPU work.

## 8. Execution and transfer scheduling

Infer owns request order and cancellation. The execution planner produces a dependency-aware sequence of compute and transfer operations.

~~~mermaid
sequenceDiagram
    participant S as Request Scheduler
    participant P as Execution Planner
    participant M as Memory Coordinator
    participant T as Transfer Scheduler
    participant G as ggml/HIP Runtime
    S->>P: next logical execution segment
    P->>M: reserve required resources
    M->>T: prefetch missing resources
    T->>G: enqueue async copy
    G-->>T: completion event/ticket
    T-->>M: resource resident
    M-->>P: Device lease
    P->>G: submit graph with dependencies
    G-->>P: execution completion fence
    P->>M: release leases after fence
    M->>M: consider safe eviction
~~~

Correctness does not depend on overlap. The first implementation may serialize execution segments while using asynchronous transfer completion. Later, prefetch for segment N+1 may overlap compute for segment N when dependencies and measurements justify it.

Avoid device-wide synchronization per transfer. Use backend stream/event dependencies. If async behavior is unsupported for a memory type, report it and use a safe fallback.

## 9. Reuse strategy

### 9.1 ggml / llama.cpp — compute substrate

Reuse selectively:
- ggml tensor representation, graph execution and proven quantized kernels;
- backend/device abstraction and buffer types;
- HIP backend and supported operation coverage;
- backend events and asynchronous tensor-copy facilities where available;
- model-format/tokenizer components only where ownership and license boundaries are suitable.

Infer owns:
- product-level request scheduler and generation lifecycle;
- model adapter contracts and architecture registry;
- global cross-resource memory budget;
- residency state machine, leases and resource-specific eviction;
- model-neutral execution/residency planner;
- CLI/configuration, telemetry and benchmark harness.

The ggml backend scheduler is low-level graph/device scheduling, not a replacement for Infer's request scheduler or global residency policy. Current ggml backend headers expose buffer types, host-buffer capabilities, asynchronous tensor copy and events. Confirm exact APIs against the pinned revision before binding.

### 9.2 Niko1221/Strata — behavior and algorithm reference

Study and selectively adapt:
- expert-aware VRAM cache and host expert arena;
- dynamic KV / resident window and host streaming;
- prefill chunking and memory-aware expert placement;
- PLE mmap and large host-backed lookup behavior;
- MTP/speculative path only after baseline correctness;
- AMD HIP build/device validation and diagnostics.

Strata is a distinct engine with model-specific execution and custom packing/quantization behavior. Do not copy CLI flags, layouts or benchmark claims without checking exact commit and running binary. Its AMD documentation warns that large pinned-host allocations can fail and mmap is not equivalent to RAM.

### 9.3 VecSzn/PolyStrata — implementation and operational evidence

Treat PolyStrata as a separate repository/checkout and record its exact revision. Inspect:
- strata-poly variant and supported models;
- dynamic KV behavior for the specific model/build;
- server/config layer versus native engine responsibilities;
- telemetry and benchmark instrumentation;
- patches and packaging that differ from upstream Strata.

Do not conflate PolyStrata with Niko1221/Strata or transfer flags/performance between them without verification. Local findings recorded a historical PolyStrata checkout and a running engine version that differed from the installed component; those are historical observations, not current upstream guarantees.

### 9.4 Fork policy

- Prefer pinning/linking upstream ggml components over copying large source areas.
- Maintain a small patch set with upstream commit references and rationale.
- If required custom GSQ-RCO kernels/types are not upstream, isolate them in a compatibility extension with tests and provenance.
- Do not create a permanent fork until a measured requirement cannot be met through a stable extension boundary.

## 10. Rust workspace and FFI boundary

~~~text
infer/
  Cargo.toml
  crates/
    infer-cli/
    infer-core/
    infer-model/
    infer-memory/
    infer-runtime/
    infer-ggml-sys/
    infer-ggml/
    infer-observability/
    infer-testkit/
    infer-model-qwen38/       # add after generic foundation
  native/
    ggml-bridge/              # small C ABI shim; CMake/Ninja
  docs/
    PRODUCT_REQUIREMENTS.md
    TECHNICAL_REQUIREMENTS.md
    REFERENCE_ARCHITECTURE.md
  tests/
  benchmarks/
~~~

Rules:
- Core, model contracts and memory policy remain backend-neutral.
- infer-ggml-sys contains raw FFI and broad unsafe code; infer-ggml exposes safe ownership-aware types.
- Native exceptions and Rust panics never cross the C ABI.
- Opaque handles have explicit create/destroy operations and one owner.
- Completion tickets retain buffers until GPU completion.
- Native implementation details do not leak into public core traits.
- Use trait objects at subsystem edges, concrete graph plans in hot loops.

## 11. Deployment and evolution

P0: native Ubuntu, one RX 9070 HIP device, one loaded model, one process, CLI, structured logs, local assets.

P1: local streaming HTTP API, more model adapters, alternative memory policies.

Implementation sequence:
1. Foundation: device discovery, ggml FFI, generic test model, basic graph execution and telemetry.
2. Memory core: coordinator, providers, leases, state machine, async transfer and failure recovery.
3. Paged state: generic attention KV contract, fake/reference provider and correctness tests.
4. Qwen adapter: exact metadata, hybrid attention/SSM, MoE, PLE and custom quant compatibility.
5. Optimization: expert prefetch, memory-aware planning, transfer/compute overlap and policy experiments.
6. Extensibility: second model adapter and optional additional backend after the core contracts survive a second architecture.

## 12. Architecture decision records to create

- ADR-001: pin ggml/llama.cpp revision and native build strategy.
- ADR-002: C ABI surface and Rust ownership model.
- ADR-003: logical resource identity and memory-tier semantics.
- ADR-004: global VRAM budget, reservations and safety reserve.
- ADR-005: KV page geometry and sequence semantics.
- ADR-006: transfer ticket, lease and GPU completion lifecycle.
- ADR-007: model adapter capability schema.
- ADR-008: GSQ-RCO custom tensor type support strategy.

## 13. Guardrails

- No model-family conditional logic in generic core.
- No HIP calls in adapters or request scheduler.
- No independent per-cache VRAM budgets.
- No eviction/free before GPU completion.
- No assumption that SSM state is attention KV.
- No SSD read in decode path by default.
- No universal LRU for every resource class.
- No speculative/MTP optimization before ordinary decode correctness.
- No performance claims from incomparable workloads.
- No dynamic plugin system, microservices or multi-GPU in the initial milestone.

## 14. Primary references

- [ggml backend API](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h)
- [ggml backend implementation interface](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend-impl.h)
- [llama.cpp context and graph scheduling](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.h)
- [Niko1221/Strata](https://github.com/Niko1221/Strata)
- [Strata engine details](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)
- [Strata AMD HIP backend](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md)
- [Strata benchmark methodology](https://github.com/Niko1221/Strata/blob/main/docs/COMMUNITY_BENCHMARKS.md)
- [VecSzn/PolyStrata](https://github.com/VecSzn/PolyStrata)
- LocalAMDAI project source notes: Qwen3.8-Flash-Next long-context findings and inference benchmark context, dated 2026-10-26. These remain separate project-context documents until intentionally copied into this repository.

All main-branch URLs are mutable. Pin and inspect a commit before implementation.