from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models


class QdrantVectorStore:
    def __init__(self, url: str, collection_name: str, dimension: int, api_key: str | None = None):
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )

    def find_similar(self, vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=max(limit, 1),
            with_payload=True,
            with_vectors=False,
        )
        matches: list[dict[str, Any]] = []
        for point in response.points:
            matches.append({
                "id": str(point.id),
                "score": float(point.score),
                "payload": point.payload or {},
            })
        return matches

    def upsert(self, item_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        try:
            point_id = str(uuid.UUID(str(item_id)))
        except (ValueError, AttributeError):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(item_id)))

        # Ensure item_id is always present in payload
        payload_copy = dict(payload)
        payload_copy.setdefault("item_id", str(item_id))

        self.client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload_copy)],
            wait=True,
        )

    def close(self) -> None:
        self.client.close()
