# Architecture

## End-to-end system

```mermaid
flowchart LR
    subgraph Offline["Offline: training pipeline (vdd CLI, PyTorch)"]
        direction TB
        RAW["data/raw/<br/>normal/ · defective/&lt;type&gt;/<br/>(+ optional masks)"] --> MAN["Manifest<br/>integrity · sha256 · 256-bit dHash<br/>image stats"]
        MAN --> EDA["EDA report<br/>imbalance · sizes · duplicates<br/>shortcut check · defect size"]
        MAN --> SPLIT["Split 65/15/20<br/>stratified by class×defect type<br/>duplicate groups kept together"]
        SPLIT --> TRAIN["Two-stage transfer learning<br/>head warm-up → full fine-tune<br/>pos_weight BCE · frozen BN stats<br/>early stop on val AP"]
        TRAIN --> THR["Threshold tuning on val<br/>F2 (recall-weighted), max-margin"]
        THR --> EXP["ONNX export<br/>+ numeric parity check"]
        EXP --> BUNDLE[("Model bundle<br/>model.onnx + metadata.json<br/>threshold · preprocessing · sha256")]
        BUNDLE --> EVAL["Test evaluation via<br/>production predictor<br/>bootstrap CIs · per-type recall"]
        EVAL --> ERR["Error analysis<br/>Grad-CAM · mask pointing game<br/>robustness · unseen-defect hold-out"]
    end

    subgraph Online["Online: inference service (Docker, no torch)"]
        direction TB
        CLIENT["Line camera / MES client"] -->|"POST /v1/predict<br/>multipart image"| MW["Middleware<br/>request-id · size guard · JSON logs"]
        MW --> VAL["Validation<br/>content-type · size · decode<br/>bomb guard · min side"]
        VAL --> SEM["Backpressure<br/>bounded semaphore → 503"]
        SEM --> PRED["DefectPredictor<br/>shared preprocess → ONNX Runtime"]
        PRED --> RESP["predicted_class · confidence<br/>P(defective) · review flag · timing"]
        PRED --> MET["/metrics (Prometheus)<br/>latency · class mix · score histogram"]
    end

    BUNDLE -->|"baked into image<br/>or mounted volume"| PRED
```

## Key design decisions

| Concern | Decision | Why |
|---|---|---|
| Train/serve skew | One framework-free `preprocess()` used by training, evaluation and API | Resize / normalisation can never diverge between offline metrics and production |
| Deployed artifact = evaluated artifact | Test metrics are computed by running the **exported ONNX bundle** through `DefectPredictor` | Numbers in the README describe what is actually served |
| Operating point | Threshold stored in `metadata.json`, overridable via `VDD_THRESHOLD_OVERRIDE` | Changing strictness for a line is a config change, versioned with the model |
| Model integrity | sha256 of `model.onnx` checked at startup; service fails fast | A corrupted or mismatched bundle never serves traffic |
| Image size | onnxruntime + FastAPI only (no torch) | ~10x smaller image, faster cold start, smaller attack surface |
| Scalability | Stateless process, 1 worker/container, ORT threads capped, horizontal replicas (HPA on CPU) | Predictable per-pod latency; linear scale-out; consistent metrics |
| Overload | Bounded in-flight inferences, 503 + `Retry-After` after `queue_timeout_s` | Fails fast instead of unbounded latency growth |
| Uncertain cases | `review_recommended` when \|p − t\| < margin | Human-in-the-loop for borderline parts instead of a silent hard decision |
| Observability | JSON logs with request id; Prometheus counters/histograms incl. score distribution | Score-distribution shift is an early, label-free drift signal |

## Request lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI (event loop)
    participant T as Threadpool
    participant O as ONNX Runtime
    C->>A: POST /v1/predict (image)
    A->>A: request-id, content-length guard, content-type check
    A->>T: decode + validate (PIL, bomb guard)
    T-->>A: RGB image / 422
    A->>A: acquire semaphore (≤ queue_timeout_s) or 503
    A->>T: preprocess + session.run
    T->>O: (1,3,224,224) float32
    O-->>T: P(defective)
    T-->>A: Prediction + timings
    A-->>C: 200 JSON (class, confidence, probability, threshold, review flag)
```
