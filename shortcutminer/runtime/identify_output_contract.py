"""Deterministic provider instructions derived from the executable Identify schema."""
import json

from shortcutminer.protocol.identify import IdentifyAgentDraft


def compact_identify_output_schema() -> dict:
    schema = IdentifyAgentDraft.model_json_schema()
    definitions = schema.get("$defs", {})
    def compact(node):
        if "$ref" in node:
            return compact(definitions[node["$ref"].removeprefix("#/$defs/")])
        result = {}
        for key in ("type", "required", "enum", "additionalProperties", "minLength", "minItems"):
            if key in node:
                result[key] = node[key]
        if "properties" in node:
            result["properties"] = {name: compact(value) for name, value in node["properties"].items()}
        if "items" in node:
            result["items"] = compact(node["items"])
        if "anyOf" in node:
            result["anyOf"] = [compact(value) for value in node["anyOf"]]
        return result
    return compact(schema)


def identify_output_contract() -> str:
    schema = json.dumps(compact_identify_output_schema(), ensure_ascii=False, separators=(",", ":"))
    return (
        "## Authoritative Identify Output Contract\n"
        "Derived deterministically from IdentifyAgentDraft.model_json_schema().\n"
        "For the final response, return exactly one IdentifyAgentDraft JSON object, not the schema.\n"
        "Use exactly the enum literals listed below; do not invent family, evidence-source, or priority names.\n"
        "Every candidate must contain all required fields, including cue.family, cue.name, description, "
        "observation, hypothesis, evidence_sources, and priority. Do not omit description, even for later candidates.\n"
        "Emit all listed properties. metadata_field_ref may be null; additional_limitations may be []. "
        "Include candidates and additional_limitations at the top level.\n"
        "This contract specifies output structure only; preserve the original scientific evidence and interpretation.\n"
        "```json\n" + schema + "\n```\n"
        "Return JSON only, without Markdown fences or commentary. "
        "The first non-whitespace character must be `{` and the last must be `}`."
    )
