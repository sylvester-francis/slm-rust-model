"""Training and evaluation for RustMentor models."""

from rustmentor.training.trainer import train_model
from rustmentor.training.evaluation import evaluate_model

__all__ = ["train_model", "evaluate_model"]
