import json
import os
from datetime import datetime

from ingestion.components.loading import Loading
from ingestion.entity.config_entity import LoadingConfig
from ingestion.constants import ARTIFACT_DIR


class LoadingPipeline:
    def __init__(self, config: LoadingConfig | None = None):
        self.config = config or LoadingConfig()

    def run(self) -> None:
        print("[LoadingPipeline] Exporting tables from PostgreSQL to Azure ...")
        artifact = Loading(self.config).run()
        print(f"  Uploaded {len(artifact.uploaded_blobs)} blobs: {artifact.uploaded_blobs}")

        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        manifest_path = os.path.join(ARTIFACT_DIR, ".loading_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(
                {"blobs": artifact.uploaded_blobs, "timestamp": str(datetime.utcnow())},
                f,
                indent=2,
            )
        print(f"  Manifest written → {manifest_path}")
        print("[LoadingPipeline] Done.\n")


if __name__ == "__main__":
    LoadingPipeline().run()
