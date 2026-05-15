from src.pipeline import TrainingPipeline

pipeline = TrainingPipeline()
content_artifact, user_artifact = pipeline.run()
