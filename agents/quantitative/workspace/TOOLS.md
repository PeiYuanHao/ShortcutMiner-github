# Quantitative Tool Policy

`tool_runtime.tools` is the live authority for available tools and input schemas. `tool_runtime.results` is the live authority for execution history. When the execution requirement is true and no current-task experiment result exists, invoke `run_experiment` using its supplied schema before finalizing.

Do not hardcode an experiment parameter schema. Do not fabricate ToolResults, reuse another task's result, cite failed calls as evidence, or report copied metric values. The finalizer owns metrics and artifacts from selected successful results. Runtime permission, validation, timeout, and budget failures must remain limitations rather than scientific findings.

Python deterministically derives evidence maturity and design strength from completed authoritative experiment measurements. The Provider does not emit these canonical fields.
