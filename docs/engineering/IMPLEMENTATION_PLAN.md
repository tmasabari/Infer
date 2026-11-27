 > **Scope note:** This engineering plan describes the separate RX 9070 / Qwen3.8 / ROCm-HIP Infer product. It is not the implementation authority for the CPU-only `BareMetalInfer` workstream. On the `BareMetalInfer` branch, use [`docs/baremetal/ARCHITECTURE_COMPARISON_AND_DECISION.md`](../baremetal/ARCHITECTURE_COMPARISON_AND_DECISION.md), [`BAREMETAL_INFER_ARCHITECTURE.md`](../baremetal/BAREMETAL_INFER_ARCHITECTURE.md), the bare-metal roadmap and work packages as the authoritative context. Keep the two projects' hardware, runtime and acceptance criteria separate.

# Infer — Engineering Execution Specification

**Status:** Proposed implementation baseline  
**Target:** Single AMD Radeon RX 9070 (16 GB), Qwen3.8, native Linux + ROCm/HIP  
**Implementation language:** Rust  
**Primary objective:** A modular local inference runtime with tiered memory and asynchronous KV streaming.

## 1. Engineering principles

1. Infer owns request orchestration, model/runtime adapters, memory policy, scheduling, observability and public APIs.
2. Reuse established tensor, graph and kernel execution primitives through a narrow backend boundary. Do not fork or reimplement kernels without measured need.
3. Keep model-specific behavior behind architecture adapters. Keep hardware-specific behavior behind backend adapters.
4. Keep memory policy independent of the physical storage/transfer implementation.
5. Start with one GPU and one model. Do not introduce multi-GPU, distributed execution, speculative decoding, quantization formats or training features into the first vertical slice unless required by Qwen3.8 correctness.
6. Every optimization must have a correctness test and a benchmark demonstrating benefit.
7. Prefer explicit ownership, bounded queues, cancellation, backpressure and deterministic resource cleanup.

## 2. Target architecture

```text
CLI / HTTP API
    |
    v
Inference Service ---- Request / Session / Sampling
    |
    v
Scheduler ------------ admission, batching, cancellation, backpressure
    |
    v
Model Runtime -------- architecture adapter + model metadata
    |                  Qwen3.8 adapter (first)
    v
Execution Backend ---- tensor/graph execution abstraction
    |                  ggml/llama.cpp HIP adapter (initial)
    v
Memory Coordinator --- global budget + placement + eviction
    |        |          |
    |        |          +---- NVMe tier (later / spill)
    |        +--------------- host RAM / pinned RAM
    +------------------------ GPU VRAM
    |
    v
Telemetry / Diagnostics / Benchmarking
```

The model adapter describes model topology and execution requirements. The backend executes operations. The memory coordinator decides placement and residency. The scheduler decides when work runs. None of these components should reach into another component's concrete implementation.

### 2.1 Core contracts

Use small traits. Avoid a universal abstraction with dozens of methods.

```rust
pub trait ModelAdapter: Send + Sync {
    fn architecture(&self) -> ArchitectureId;
    fn inspect(&self, source: &ModelSource) -> Result<ModelDescriptor>;
    fn create_runtime(
        &self,
        model: &ModelDescriptor,
        backend: Arc<dyn ExecutionBackend>,
        memory: Arc<dyn MemoryManager>,
    ) -> Result<Box<dyn ModelRuntime>>;
}

pub trait ModelRuntime: Send {
    fn descriptor(&self) -> &ModelDescriptor;
    fn prefill(&mut self, input: TokenBatch, state: &mut SequenceState)
        -> Result<PrefillOutput>;
    fn decode(&mut self, input: DecodeInput, state: &mut SequenceState)
        -> Result<DecodeOutput>;
}

pub trait ExecutionBackend: Send + Sync {
    fn capabilities(&self) -> BackendCapabilities;
    fn allocate(&self, request: AllocationRequest) -> Result<BackendBuffer>;
    fn copy(&self, request: CopyRequest) -> Result<TransferHandle>;
    fn synchronize(&self, scope: SyncScope) -> Result<()>;
}

pub trait MemoryManager: Send + Sync {
    fn reserve(&self, request: MemoryRequest) -> Result<MemoryLease>;
    fn release(&self, lease: MemoryLeaseId) -> Result<()>;
    fn residency(&self, object: MemoryObjectId) -> Result<Residency>;
    fn request_migration(&self, request: MigrationRequest)
        -> Result<TransferHandle>;
}

pub trait KvStore: Send + Sync {
    fn create_sequence(&self, request: KvSequenceRequest) -> Result<KvSequenceId>;
    fn append(&self, request: KvAppendRequest) -> Result<()>;
    fn read(&self, request: KvReadRequest) -> Result<KvReadHandle>;
    fn truncate(&self, request: KvTruncateRequest) -> Result<()>;
    fn remove_sequence(&self, id: KvSequenceId) -> Result<()>;
}

pub trait Scheduler: Send + Sync {
    fn submit(&self, request: InferenceRequest) -> Result<RequestHandle>;
    fn cancel(&self, id: RequestId) -> Result<()>;
}
```

These are design contracts, not a mandate to implement all methods in the first commit. Refine signatures against concrete ownership and async requirements before freezing the public API. Use typed IDs and request structs; avoid passing raw device pointers across layers.

### 2.2 Dependency direction

- Domain types depend on the Rust standard library and small foundational crates only.
- Application services depend on domain traits, never concrete backend implementations.
- Adapters implement domain ports.
- Composition root wires concrete implementations.
- Tests use in-memory fakes for ports and hardware integration tests for the HIP adapter.

## 3. Proposed repository layout

```text
Cargo.toml                         # workspace
crates/
  infer-core/                      # IDs, errors, config, model/memory/request types
  infer-application/               # inference service, orchestration, scheduler ports
  infer-models/                    # adapter registry and model adapters
  infer-model-qwen/                 # Qwen3.8 architecture adapter
  infer-memory/                    # memory coordinator, placement, eviction, KV policy
  infer-backend-api/                # execution backend contracts and shared types
  infer-backend-ggml/               # narrow ggml/llama.cpp FFI adapter
  infer-backend-hip/                # HIP device discovery, buffers, transfers
  infer-io/                         # model file loading and tokenizer integration
  infer-server/                     # CLI / HTTP surface and composition root
  infer-observability/              # metrics, tracing, diagnostics
  infer-bench/                      # repeatable benchmark harness
tests/
  fixtures/
  contract/
  integration/
docs/
  architecture/
  decisions/
  engineering/
```

Keep workspace crate count lean if implementation reveals that two packages have no independent reason to exist. Do not split crates merely to satisfy a diagram.

## 4. Work packages

### WP-00 — Repository and build foundation

**Goal:** Establish a reproducible Rust workspace and CI baseline.

**Files**
- `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`
- `crates/infer-core/Cargo.toml`, `crates/infer-core/src/lib.rs`
- `.github/workflows/ci.yml`
- `rustfmt.toml`, `clippy.toml` (only where needed)
- `docs/decisions/0001-workspace-and-toolchain.md`

**Interfaces**
- Workspace-level crate boundaries only; no runtime APIs yet.
- Define shared `Result<T>`, typed error categories and stable IDs in infer-core.

**Dependencies:** None.

**Implementation tasks**
1. Create workspace and minimal library/binary crate skeletons.
2. Pin supported Rust toolchain and document Linux build prerequisites.
3. Add format, lint, unit-test and documentation CI jobs.
4. Define error policy: contextual errors at boundaries; no panic for runtime failures.
5. Add feature flags for optional HIP integration; default build must work without ROCm installed.

**Acceptance tests**
- `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` pass in CI.
- Default workspace builds on Linux without ROCm libraries.
- HIP-enabled build fails with a clear prerequisite diagnostic when ROCm is absent, not an opaque linker error.
- No cyclic crate dependencies.

### WP-01 — Core domain contracts and configuration

**Goal:** Define stable, backend-neutral concepts used by all packages.

**Files**
- `crates/infer-core/src/{error,ids,config,model,memory,kv,request,token}.rs`
- `crates/infer-core/src/lib.rs`
- `docs/architecture/domain-model.md`

**Interfaces**
- `ModelDescriptor`, `ArchitectureId`, `ModelSource`
- `InferenceRequest`, `RequestId`, `SequenceState`
- `MemoryTier`, `MemoryObjectId`, `MemoryLease`, `Residency`
- `KvSequenceId`, `KvBlockId`, `KvLayout`
- `InferConfig`, `DeviceBudget`, `RuntimeLimits`

**Dependencies:** WP-00.

**Implementation tasks**
1. Model memory tiers as explicit ordered capabilities: device VRAM, pinned host memory, pageable host RAM, optional NVMe.
2. Separate logical KV identity and layout from physical allocation.
3. Define config validation and defaults without hardcoding RX 9070 assumptions into domain types.
4. Define error categories for model, backend, memory pressure, transfer, cancellation and invalid configuration.
5. Use byte counts and token counts as distinct newtypes to prevent unit confusion.

**Acceptance tests**
- Serialization round-trips for supported configuration.
- Invalid budgets, negative-equivalent values and incompatible limits are rejected.
- Domain types compile and test without HIP, ggml or ROCm dependencies.
- Unit tests prove IDs and units cannot be accidentally interchanged.

### WP-02 — Adapter registry and model loading

**Goal:** Discover model architecture and load tokenizer/model metadata through replaceable adapters.

**Files**
- `crates/infer-models/src/{lib,registry,loader}.rs`
- `crates/infer-io/src/{lib,model_source,tokenizer}.rs`
- `crates/infer-application/src/ports/model.rs`
- `tests/contract/model_adapter.rs`

**Interfaces**
- `ModelAdapter`, `ModelRegistry`, `ModelLoader`, `Tokenizer`
- Registry lookup by explicit architecture identifier and validated metadata; no arbitrary dynamic code loading.

**Dependencies:** WP-01.

**Implementation tasks**
1. Implement adapter registration and deterministic lookup.
2. Implement model source abstraction for local files/directories.
3. Validate metadata before allocating GPU memory.
4. Keep tokenizer loading behind its own port.
5. Return actionable errors for unsupported architecture, tensor type and metadata version.

**Acceptance tests**
- Fake adapters can be registered and selected without backend dependencies.
- Duplicate architecture registrations fail deterministically.
- Unsupported models fail before device allocation.
- Malformed metadata and missing tokenizer assets produce actionable errors.

### WP-03 — Execution backend API and ggml bridge

**Goal:** Execute model operations through a narrow, replaceable backend boundary.

**Files**
- `crates/infer-backend-api/src/{lib,backend,buffer,transfer,capabilities}.rs`
- `crates/infer-backend-ggml/src/{lib,ffi,context,graph,buffer}.rs`
- `crates/infer-backend-ggml/build.rs`
- `docs/decisions/0002-ggml-ffi-boundary.md`

**Interfaces**
- `ExecutionBackend`, `BackendBuffer`, `TransferHandle`, `BackendCapabilities`
- Opaque handles across FFI; explicit ownership and destruction functions.
- No ggml types in infer-core or application public APIs.

**Dependencies:** WP-00, WP-01. HIP adapter work in WP-04 can proceed in parallel against backend API contracts.

**Implementation tasks**
1. Select and pin the ggml/llama.cpp revision after a small compatibility spike.
2. Wrap only the operations required for model load, tensor allocation, graph execution and transfer.
3. Define safe Rust ownership around C handles and callback lifetimes.
4. Keep C/C++ glue minimal and isolate unsafe blocks.
5. Add a backend capability query; do not infer capabilities from GPU names.

**Acceptance tests**
- Fake backend passes backend contract tests.
- FFI handles are released exactly once on success and error paths.
- No raw FFI pointer escapes the adapter crate.
- A small deterministic tensor operation matches a CPU reference within defined tolerance.
- Backend can be replaced in application tests without changing application code.

### WP-04 — HIP device and transfer adapter

**Goal:** Provide RX 9070 discovery, device allocations and asynchronous host/device transfers.

**Files**
- `crates/infer-backend-hip/src/{lib,device,allocation,stream,event,transfer}.rs`
- `crates/infer-backend-hip/build.rs`
- `tests/integration/hip_device.rs`
- `docs/operations/rocm-linux.md`

**Interfaces**
- Implements backend device and transfer ports.
- `DeviceInfo` exposes stable capability data: total/free memory, architecture, transfer support and backend version.
- Transfer handles expose completion/error state without leaking HIP event types.

**Dependencies:** WP-03 contracts; can develop in parallel with WP-03 implementation after contract review.

**Implementation tasks**
1. Enumerate devices and identify the selected GPU by explicit configuration.
2. Validate RX 9070/gfx1201 and ROCm runtime compatibility at startup.
3. Implement device buffer allocation/free and bounded asynchronous copy queues.
4. Implement pinned host allocation and event-based transfer completion.
5. Add explicit stream ownership and synchronization scopes.
6. Handle device loss, allocation failure and shutdown safely.

**Acceptance tests**
- Unit tests use a fake HIP facade for allocation and transfer error paths.
- Hardware test identifies the RX 9070 and reports correct device properties.
- Host-to-device and device-to-host round-trip tests preserve bytes.
- Transfer completion is event-driven; no unconditional device-wide synchronization in the hot path.
- Repeated allocate/copy/free cycles show no unbounded VRAM growth.
- Tests skip with an explicit reason on machines without a supported HIP device.

### WP-05 — Tiered memory coordinator

**Goal:** Enforce one global memory budget and place objects across VRAM, pinned RAM, pageable RAM and (later) NVMe.

**Files**
- `crates/infer-memory/src/{lib,coordinator,budget,allocator,placement,eviction}.rs`
- `crates/infer-memory/src/policy/{mod,kv_policy,weight_policy}.rs`
- `tests/contract/memory_manager.rs`
- `tests/unit/memory_pressure.rs`

**Interfaces**
- `MemoryManager`, `MemoryPolicy`, `TierAllocator`, `EvictionPolicy`
- Memory objects carry size, purpose, priority, last-use metadata and current residency.
- All allocations go through the coordinator; no component independently consumes the full VRAM budget.

**Dependencies:** WP-01, WP-03, WP-04.

**Implementation tasks**
1. Reserve VRAM headroom for runtime, kernels, graph workspaces and fragmentation.
2. Track live allocations and leases centrally.
3. Implement admission control before allocation.
4. Implement deterministic LRU/priority eviction as the initial policy.
5. Support asynchronous migration with bounded in-flight bytes and backpressure.
6. Keep KV and model-weight policies separate while sharing the same global budget.
7. Implement pageable RAM and NVMe as later tiers; first vertical slice may use VRAM + pinned host RAM only.

**Acceptance tests**
- Global budget is never exceeded in simulated concurrent allocation tests.
- Failed reservations leave no leaked lease or partial accounting.
- Pinned-memory limits are enforced independently from pageable RAM.
- Eviction never removes leased/in-use buffers.
- Concurrent migration is bounded and cancellation-safe.
- Stress tests show stable accounting after repeated allocation, eviction and release.

### WP-06 — KV block store and streaming

**Goal:** Store KV by logical blocks and stream blocks between VRAM and host memory without coupling the model adapter to physical placement.

**Files**
- `crates/infer-memory/src/kv/{mod,store,block,layout,streamer,prefetch}.rs`
- `crates/infer-application/src/ports/kv.rs`
- `tests/unit/kv_store.rs`
- `tests/integration/kv_streaming.rs`
- `docs/architecture/kv-streaming.md`

**Interfaces**
- `KvStore`, `KvBlockStore`, `KvStreamer`, `KvReadHandle`
- Operations: create sequence, append block, read range, prefetch, truncate and release.
- Logical block references remain stable while physical residency changes.

**Dependencies:** WP-01, WP-04, WP-05.

**Implementation tasks**
1. Define KV block geometry from model layer/head/dtype metadata; do not hardcode Qwen dimensions in the store.
2. Allocate recent/hot blocks in VRAM and eligible cold blocks in pinned host memory.
3. Stream required blocks asynchronously and prefetch the next expected blocks.
4. Preserve ordering and sequence isolation across concurrent requests.
5. Implement bounded transfer queues, backpressure and cancellation.
6. Ensure a block is not evicted while a decode operation holds a read lease.
7. Add instrumentation for bytes moved, transfer latency, cache hits and stalls.

**Acceptance tests**
- Append/read/truncate correctness across block boundaries.
- Byte-for-byte equality after VRAM → pinned RAM → VRAM round trip.
- Concurrent sequences never read one another's KV.
- Cancellation during migration leaves the store consistent.
- A decode read cannot observe a partially transferred block.
- Long-context test demonstrates bounded VRAM residency while total KV exceeds the configured VRAM KV budget.
- Benchmark reports streaming overhead against an all-resident baseline; no performance claim is accepted without this comparison.

### WP-07 — Qwen3.8 architecture adapter

**Goal:** Run the initial target model using generic engine contracts.

**Files**
- `crates/infer-model-qwen/src/{lib,adapter,metadata,topology,runtime,kv_layout}.rs`
- `tests/fixtures/qwen/`
- `tests/contract/qwen_adapter.rs`
- `tests/integration/qwen_smoke.rs`

**Interfaces**
- Implements `ModelAdapter` and `ModelRuntime`.
- Exposes architecture metadata, tensor mapping, layer execution plan and KV layout through typed descriptors.
- Does not own GPU allocation or transfer policy.

**Dependencies:** WP-02, WP-03, WP-05, WP-06.

**Implementation tasks**
1. Confirm exact Qwen3.8 model variant, GGUF metadata and supported tensor/quantization formats.
2. Map model metadata to generic model descriptor and backend tensor representation.
3. Implement prefill and single-token decode paths.
4. Connect KV reads/writes through KvStore contracts.
5. Validate logits and token outputs against a trusted reference runtime.
6. Reject unsupported architecture features explicitly rather than silently approximating them.

**Acceptance tests**
- Metadata parser handles valid fixture and rejects malformed/incompatible files.
- Small prompt prefill and decode produce finite logits.
- Token output matches reference runtime under defined deterministic settings/tolerance.
- KV cache on/off or all-resident/streamed paths produce equivalent outputs within numerical tolerance.
- Model adapter tests run against fake backend and fake memory manager.

### WP-08 — Scheduler and inference service

**Goal:** Provide request lifecycle, cancellation and controlled concurrency.

**Files**
- `crates/infer-application/src/{lib,service,scheduler,request_state,session}.rs`
- `crates/infer-application/src/ports/{model,backend,kv}.rs`
- `tests/unit/scheduler.rs`
- `tests/integration/request_lifecycle.rs`

**Interfaces**
- `InferenceService::generate`, `Scheduler::submit/cancel`, `RequestHandle`
- Request states: queued, loading, prefill, decoding, completed, cancelled, failed.
- Streaming token output via bounded channel; consumer backpressure must propagate to generation.

**Dependencies:** WP-02, WP-05, WP-06, WP-07.

**Implementation tasks**
1. Implement single-request execution first.
2. Add bounded admission queue and explicit overload response.
3. Implement cancellation and cleanup for model, KV and memory leases.
4. Add token streaming with bounded channel.
5. Keep continuous batching as a later optimization after single-request correctness and measurements.

**Acceptance tests**
- End-to-end request returns tokens and a terminal status.
- Cancellation releases KV and memory leases.
- Slow consumers cause bounded backpressure, not unbounded memory growth.
- Queue saturation returns deterministic overload behavior.
- Failure in one request does not corrupt another request's state.

### WP-09 — API, CLI and composition root

**Goal:** Make the engine usable without leaking internal architecture.

**Files**
- `crates/infer-server/src/{main,bootstrap,config,cli,http}.rs`
- `crates/infer-server/Cargo.toml`
- `docs/operations/quickstart.md`

**Interfaces**
- CLI commands: `devices`, `inspect-model`, `serve`, `benchmark`.
- Initial HTTP endpoints: health, model metadata, generation (streaming), cancellation.
- Composition root selects model adapter, backend, memory policy and scheduler from validated configuration.

**Dependencies:** WP-04, WP-07, WP-08.

**Implementation tasks**
1. Wire concrete components in one composition root.
2. Add explicit device/model/config selection.
3. Implement health and diagnostics endpoints.
4. Implement streaming generation and cancellation.
5. Document startup checks and common ROCm errors.

**Acceptance tests**
- CLI help and inspect-model work without GPU access.
- Device command reports selected GPU and backend capability.
- Server rejects invalid configuration before loading model.
- Streaming endpoint emits ordered tokens and terminal status.
- Cancellation endpoint stops the matching request only.

### WP-10 — Telemetry, benchmark and performance gates

**Goal:** Make performance and memory behavior measurable and reproducible.

**Files**
- `crates/infer-observability/src/{lib,metrics,tracing,diagnostics}.rs`
- `crates/infer-bench/src/{main,scenarios,report}.rs`
- `benchmarks/baselines/README.md`
- `docs/performance/benchmark-methodology.md`

**Metrics**
- Prompt tokens/sec, decode tokens/sec, time-to-first-token.
- GPU VRAM peak/current, host/pinned memory, KV bytes by tier.
- Transfer bytes, transfer latency, queue depth, migration stalls and evictions.
- Request latency, cancellation latency and errors.

**Dependencies:** WP-04 through WP-09; basic instrumentation should be added incrementally in earlier packages.

**Implementation tasks**
1. Add low-overhead counters and histograms around allocation, transfer and token generation.
2. Define repeatable warmup, prompt, context and decode workloads.
3. Capture hardware/software/model/quantization/config metadata with every result.
4. Compare streaming enabled/disabled at equivalent settings.
5. Establish regression thresholds only after stable baselines exist.

**Acceptance tests**
- Metrics are internally consistent with allocation and transfer accounting.
- Benchmark output is machine-readable and includes complete environment metadata.
- Scenarios cover short context and long context, prefill and decode separately.
- Benchmark can detect deliberate transfer or decode regressions.
- No performance threshold is declared until repeated baseline runs establish variance.

## 5. Dependency graph and delivery sequence

```text
WP-00
  |
WP-01
  |-------------------|
WP-02              WP-03
  |                   |
  |                WP-04
  |                   |
  |                WP-05 <---+
  |                   |      |
  +-------------------+      |
                      WP-06 -+
                        |
                      WP-07
                        |
                      WP-08
                        |
                      WP-09

WP-10 instrumentation begins in WP-04 and completes after WP-09.
```

Recommended milestones:

- **M0 — Buildable foundation:** WP-00, WP-01.
- **M1 — Backend vertical slice:** WP-02, WP-03, WP-04; load and execute a deterministic tensor operation.
- **M2 — Memory foundation:** WP-05; enforce global VRAM and pinned-memory budgets.
- **M3 — KV streaming proof:** WP-06; prove correctness and bounded VRAM residency with synthetic KV.
- **M4 — First model:** WP-07; Qwen3.8 prefill/decode with all-resident KV.
- **M5 — Integrated inference:** WP-08, WP-09; generate and stream tokens through the service.
- **M6 — Long-context optimization:** streamed KV end-to-end, benchmarked against baseline; WP-10.

The schedule is dependency-driven, not calendar-driven. WP-03 and WP-04 may overlap after backend contracts are reviewed. WP-06 synthetic KV testing should not wait for Qwen3.8.

## 6. Cross-cutting quality gates

Every work package must include:
- Unit tests for domain logic and failure paths.
- Contract tests for every trait implementation.
- Integration tests at external boundaries.
- Structured errors with actionable context.
- Explicit resource ownership and cleanup.
- Documentation of configuration and unsupported cases.
- No hidden global mutable state.
- No backend-specific types in domain/application APIs.
- No unbounded queues or implicit unbounded buffering.

### Performance acceptance protocol

Record the current trusted runtime baseline on the same RX 9070, model file, quantization, prompt, context and generation length. Run at least:
- Contexts: 256, 8K, 32K, 64K and 128K where model/runtime support permits.
- Prefill and decode separately.
- Streaming off/on with identical sampling and output length.
- Peak VRAM, host RAM, pinned RAM, bytes transferred and stall time.
- At least three measured repetitions after warmup; report median and spread.

Do not make a throughput target a hard acceptance gate until a reproducible baseline and variance are recorded. Correctness, bounded memory and no resource leaks are hard gates from the beginning.

## 7. Explicitly deferred scope

Unless a measured requirement changes priority, defer:
- Multi-GPU and distributed inference.
- Windows runtime support (keep interfaces portable, but validate native Linux first).
- Training and fine-tuning.
- Speculative decoding and advanced batching.
- NVMe KV paging in the first working vertical slice.
- Custom kernels and custom quantization formats.
- Automatic model downloading and arbitrary plugin code execution.

## 8. Architecture decision records required before implementation

1. Exact Qwen3.8 model variant, file format, quantization and tokenizer.
2. ggml/llama.cpp revision and exact FFI surface.
3. HIP/ROCm version and supported RX 9070/gfx1201 matrix.
4. KV block layout, alignment, dtype and transfer granularity.
5. VRAM reserve policy and pinned-memory budget.
6. Async runtime choice and ownership model for HIP streams/events.
7. Numerical equivalence tolerances against the reference runtime.

## 9. Definition of done for the first usable release

A user can select the RX 9070, load the supported Qwen3.8 model, submit a prompt, receive streamed tokens, and run a context length whose KV footprint exceeds the configured VRAM KV budget. KV blocks migrate between VRAM and pinned host memory without corrupting output; VRAM usage remains within the global budget; cancellation and shutdown release resources; and benchmark reports quantify the cost of streaming against an all-resident baseline.
