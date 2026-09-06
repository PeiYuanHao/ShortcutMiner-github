"""Small typed interface shared by all registered tools."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from shortcutminer.tooling.protocol import AgentName, ToolSpec


@dataclass(frozen=True)
class ToolExecutionContext:
    run_id: str
    agent_name: AgentName
    dataset_root: Path | None = None
    runs_root: Path | None = None
    workspace_root: Path | None = None
    approved_artifacts: frozenset[str] = field(default_factory=frozenset)
    approved_samples: frozenset[str] = field(default_factory=frozenset)
    max_selected_samples: int = 100
    tool_schemas: tuple[dict[str, Any], ...] = ()


class Tool(Protocol):
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]

    def execute(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> BaseModel: ...

    def spec(self) -> ToolSpec: ...


ToolHandler = Callable[[BaseModel, ToolExecutionContext], BaseModel]


@dataclass(frozen=True)
class StructuredTool:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: ToolHandler

    def execute(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> BaseModel:
        return self.handler(arguments, context)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
            output_schema=self.output_model.model_json_schema(),
        )
