from pathlib import Path

from pydantic import BaseModel, Field

from shortcutminer.protocol.identify import IdentifyRequest
from tools.common.dataset_inspector import inspect_dataset
from tools.common.metadata_profiler import profile_metadata
from tools.common.sample_selector import select_samples


class IdentifyDiscoveryContext(BaseModel):
    protocol_version: str = "1.0"
    run_id: str

    dataset_name: str
    task_type: str
    label_column: str

    dataset_inspection: dict
    metadata_profile: dict | None = None

    selected_samples: list[dict] = Field(
        default_factory=list
    )

    limitations: list[str] = Field(
        default_factory=list
    )


def build_identify_context(
    request: IdentifyRequest,
    seed: int,
) -> IdentifyDiscoveryContext:
    """
    Build deterministic discovery context for the Identify Agent.

    This function prepares evidence only.
    It does not generate shortcut candidates.
    """

    limitations: list[str] = []

    dataset_info = inspect_dataset(
        request.dataset.root
    )

    metadata_profile = None
    selected_samples: list[dict] = []

    if not dataset_info["exists"]:
        limitations.append(
            "Dataset root is not accessible."
        )

        return IdentifyDiscoveryContext(
            run_id=request.run_id,
            dataset_name=request.dataset.name,
            task_type=request.task.type,
            label_column=request.task.label_column,
            dataset_inspection=dataset_info,
            metadata_profile=None,
            selected_samples=[],
            limitations=limitations,
        )

    if request.dataset.metadata_file:
        metadata_path = (
            Path(request.dataset.root)
            / request.dataset.metadata_file
        )

        metadata_profile = profile_metadata(
            str(metadata_path)
        )

        if not metadata_profile["exists"]:
            limitations.append(
                "Configured metadata file is not accessible."
            )

        else:
            try:
                selected_samples = select_samples(
                    metadata_path=str(metadata_path),
                    max_samples=request.budget.max_samples,
                    seed=seed,
                    strategy="class_balanced",
                    label_column=request.task.label_column,
                )

            except ValueError as exc:
                limitations.append(
                    f"Class-balanced sampling failed: {exc}"
                )

                try:
                    selected_samples = select_samples(
                        metadata_path=str(metadata_path),
                        max_samples=request.budget.max_samples,
                        seed=seed,
                        strategy="random",
                    )

                    limitations.append(
                        "Random sampling was used as fallback."
                    )

                except (ValueError, FileNotFoundError) as fallback_exc:
                    limitations.append(
                        "Sample selection failed: "
                        f"{fallback_exc}"
                    )

    else:
        limitations.append(
            "No metadata file was configured."
        )

    return IdentifyDiscoveryContext(
        run_id=request.run_id,
        dataset_name=request.dataset.name,
        task_type=request.task.type,
        label_column=request.task.label_column,
        dataset_inspection=dataset_info,
        metadata_profile=metadata_profile,
        selected_samples=selected_samples,
        limitations=limitations,
    )