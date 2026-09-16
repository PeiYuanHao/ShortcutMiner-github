# Quantitative Final Gate

If mandatory execution has not been attempted, call `run_experiment`; do not finalize. If an attempt failed, Q0 blocked/failed may be warranted. If an attempt succeeded but the effect is null, weak, or inconclusive, use completed with a justified non-Q0 level and an inconclusive direction.

For a final response, emit no prose or markdown. Return exactly one valid JSON object matching `QuantitativeAgentDraft`. The first non-whitespace character must be `{` and the last non-whitespace character must be `}`.
