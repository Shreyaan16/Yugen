from __future__ import annotations
from typing import Iterable, Optional
from src.components.loading import Loading
from src.constants import CONNECTION_STRING, CONTAINER_NAME, DATABASE_URL

def run_loading(schema: str = "public",) -> list[str]:
	with Loading(pg_connection_string=DATABASE_URL, azure_connection_string=CONNECTION_STRING,
			  azure_container_name=CONTAINER_NAME,) as loader:
		return loader.export_all_tables(schema=schema)
	
if __name__ == "__main__":
	uploaded_tables = run_loading()
	print(f"Uploaded {len(uploaded_tables)} tables to {CONTAINER_NAME}")
