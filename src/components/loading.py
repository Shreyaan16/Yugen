import io
from typing import Optional

import pandas as pd
from azure.storage.blob import BlobServiceClient, ContentSettings
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

class Loading:
    def __init__(self, pg_connection_string: str, azure_connection_string: str, azure_container_name: str):
        self.pg_connection_string = pg_connection_string
        self.azure_connection_string = azure_connection_string
        self.azure_container_name = azure_container_name
        self._engine: Optional[Engine] = None

    def connect(self) -> None:
        self._engine = create_engine(self.pg_connection_string)

        self._blob_service = BlobServiceClient.from_connection_string(self.azure_connection_string)
        container_client = self._blob_service.get_container_client(self.azure_container_name)
        if not container_client.exists():
            container_client.create_container()

    def disconnect(self) -> None:
        if self._engine is not None:
            self._engine.dispose()

    def export_table(self, table_name: str, schema: str = "public") -> str:
        self._ensure_connected()

        qualified = f'"{schema}"."{table_name}"'
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

    def export_all_tables(self, schema: str = "public",) -> list[str]:
        self._ensure_connected()

        tables = self._list_tables(schema)
        tables = [t for t in tables]

        uploaded = []
        for table in tables:
            try:
                blob_path = self.export_table(table, schema=schema)
                uploaded.append(blob_path)
            except Exception as exc:
                raise exc
            
        return uploaded

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
        blob_client = self._blob_service.get_blob_client(container=self.azure_container_name, blob=blob_name,)
        blob_client.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type="text/csv"))
