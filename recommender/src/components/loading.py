import io
from typing import Optional

import pandas as pd
from azure.storage.blob import BlobServiceClient, ContentSettings
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from recommender.src.entity.artifact_entity import LoadingArtifact
from recommender.src.entity.config_entity import LoadingConfig

class Loading:
    def __init__(self, config: LoadingConfig):
        self.config = config
        self._engine: Optional[Engine] = None
        self._blob_service: Optional[BlobServiceClient] = None

    def connect(self) -> None:
        self._engine = create_engine(self.config.pg_connection_string)

        self._blob_service = BlobServiceClient.from_connection_string(self.config.azure_connection_string)
        container_client = self._blob_service.get_container_client(self.config.azure_container_name)
        if not container_client.exists():
            container_client.create_container()

    def disconnect(self) -> None:
        if self._engine is not None:
            self._engine.dispose()

    def export_table(self, table_name: str, schema: Optional[str] = None) -> str:
        self._ensure_connected()

        target_schema = schema or self.config.schema
        qualified = f'"{target_schema}"."{table_name}"'
        blob_name = f"{table_name}.csv"

        df = pd.read_sql(f"SELECT * FROM {qualified}", self._engine)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)

        buffer.seek(0)
        self._upload_blob(blob_name, buffer.getvalue())
        return blob_name

    def export_query(self, query: str, blob_name: str) -> str:
        self._ensure_connected()

        full_blob_name = f"{blob_name}"

        df = pd.read_sql(query, self._engine)

        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        self._upload_blob(full_blob_name, buffer.getvalue())
        return full_blob_name

    def export_all_tables(self, schema: Optional[str] = None,) -> list[str]:
        self._ensure_connected()

        target_schema = schema or self.config.schema
        tables = self._list_tables(target_schema)
        tables = [t for t in tables]

        uploaded = []
        for table in tables:
            try:
                blob_path = self.export_table(table, schema=target_schema)
                uploaded.append(blob_path)
            except Exception as exc:
                raise exc

            
        return uploaded

    def run(self) -> LoadingArtifact:
        uploaded = self.export_all_tables(schema=self.config.schema)
        return LoadingArtifact(uploaded_blobs=uploaded)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False  # don't suppress exceptions

    def _ensure_connected(self) -> None:
        if self._engine is None:
            raise RuntimeError("Not connected. Call connect() or use the class as a context manager.")
        if self._blob_service is None:
            raise RuntimeError("Azure client not initialised. Call connect() first.")

    def _list_tables(self, schema: str) -> list[str]:
        query = text(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = :schema
              AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """
        )
        with self._engine.connect() as conn:
            result = conn.execute(query, {"schema": schema})
            return [row[0] for row in result.fetchall()]

    def _upload_blob(self, blob_name: str, data: str) -> None:
        blob_client = self._blob_service.get_blob_client(
            container=self.config.azure_container_name,
            blob=blob_name,
        )
        blob_client.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type="text/csv"))
