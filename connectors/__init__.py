"""Absolver compute platform connectors."""

from connectors.modal_runner import run_pipeline_modal
from connectors.molab_runner import run_pipeline_molab

__all__ = ["run_pipeline_modal", "run_pipeline_molab"]
