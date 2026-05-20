from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.inference_pipeline import InferencePipeline


def stage_training() -> None:
    TrainingPipeline().run()


__all__ = [
    "TrainingPipeline",
    "InferencePipeline",
    "stage_training",
]
