from recommender.components.evaluation import Evaluation
from recommender.entity.artifact_entity import EvaluationArtifact
from recommender.entity.config_entity import EvaluationConfig


class EvaluationPipeline:
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

    def run(self) -> EvaluationArtifact:
        evaluation = Evaluation(config=self.config)
        return evaluation.run()


if __name__ == "__main__":
    EvaluationPipeline().run()
