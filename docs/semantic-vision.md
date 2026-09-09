# Provider-backed Semantic Vision

`SemanticVisualInspector` remains an optional provider-neutral capability. `MockSemanticVisualInspector` supports deterministic tests, while `OpenAICompatibleSemanticVisualInspector` uses the hardened multimodal runtime for configured compatible endpoints.

The capability is deliberately an observer, not a sixth Agent and not a shortcut detector. It may return structured visible facts about foreground objects, background context, text or markers, acquisition artifacts, environments, accessories, and spatial context. It may not return candidates, support/contradiction verdicts, risk, causality, or hidden truth. Identify alone can combine these observations with deterministic descriptors and public dataset evidence to propose an unverified candidate.

An observation's `high`, `medium`, `low`, or `unknown` confidence qualifies only the VLM's visual description. It is not shortcut confidence, statistical strength, or a final assessment.

Only sample references already selected and decoded by Visual Discovery are passed to the inspector. Each reference is validated as a safe dataset-relative path and resolved beneath the dataset root. Before encoding, the implementation enforces file existence, per-image and total byte limits, decoded format, pixel limits, batch size, and request count. Supported encoded formats are JPEG, PNG, WebP, and GIF. Image data is carried ephemerally as data URLs and is not persisted.

The system prompt treats pixels, visible text, filenames, and metadata as untrusted evidence, never instructions. Provider JSON is validated against `SemanticObservationBatch`, and every returned sample ID/reference must belong to the current batch. Invalid JSON and schema responses use the model runtime's bounded schema retry and telemetry. A VLM failure records semantic unavailability while preserving deterministic Visual Discovery results.

Runtime artifacts distinguish `disabled`, `mock`, and `real_provider`, and real-provider status records the configured model. Declaring `supports_image_input: true` is required; an endpoint that does not support the multimodal content shape leaves semantic capability unavailable. This is OpenAI-compatible multimodal support, not a claim of universal VLM-provider compatibility or scientific shortcut-discovery accuracy.
