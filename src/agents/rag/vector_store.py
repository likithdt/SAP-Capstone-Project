"""
Vector Store for Equipment Maintenance Manuals and RAG Retrieval.
Utilizes ChromaDB for semantic vector search over industrial asset SOPs and maintenance manuals.
"""

import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings

from src.config import AgentConfig, DEFAULT_AGENT_CONFIG
from src.agents.rag.knowledge_base import EQUIPMENT_MAINTENANCE_MANUALS, MaintenanceManualDocument


class MaintenanceVectorStore:
    """
    Semantic Vector Search Store for rotating asset maintenance manuals and SOPs.
    Indexes engineering manuals and provides cosine/Euclidean nearest neighbor retrieval.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        in_memory: bool = True,
        collection_name: Optional[str] = None,
    ):
        self.config = config or DEFAULT_AGENT_CONFIG
        self.collection_name = collection_name or self.config.collection_name
        self.in_memory = in_memory

        if self.in_memory:
            self.client = chromadb.Client()
        else:
            os.makedirs(self.config.chroma_persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.config.chroma_persist_dir)

        self.collection = self._get_or_create_collection()
        if self.collection.count() == 0:
            self.seed_manuals()

    def _get_or_create_collection(self):
        """Retrieve existing collection or create new one in ChromaDB."""
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def seed_manuals(self, manuals: Optional[List[MaintenanceManualDocument]] = None) -> int:
        """
        Chunk, embed, and ingest maintenance manuals into ChromaDB collection.
        Returns the number of indexed document chunks.
        """
        docs_to_index = manuals or EQUIPMENT_MAINTENANCE_MANUALS
        ids = []
        documents = []
        metadatas = []

        for manual in docs_to_index:
            doc_id = manual.doc_id
            content_text = f"{manual.title}\n{manual.content.strip()}"
            metadata = {
                "doc_id": manual.doc_id,
                "title": manual.title,
                "sop_code": manual.sop_code,
                "fault_class": manual.fault_class,
                "fault_name": manual.fault_name,
                "category": manual.category,
                "estimated_hours": float(manual.estimated_hours),
                "work_center": str(manual.metadata.get("work_center", "MECH-01")),
                "criticality": str(manual.metadata.get("criticality", "MEDIUM")),
                "primary_material": str(manual.metadata.get("primary_material", "NONE")),
            }
            ids.append(doc_id)
            documents.append(content_text)
            metadatas.append(metadata)

        # Upsert documents into collection
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

    def search(
        self,
        query: str,
        n_results: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity vector search over maintenance manuals.

        Args:
            query: Semantic text query (e.g. 'bearing spalling 107.4 Hz kurtosis')
            n_results: Maximum number of manual chunks to retrieve (defaults to config.rag_top_k)
            filter_metadata: Optional metadata filter dict e.g. {'fault_class': 1}

        Returns:
            List of retrieved manuals with document text, metadata, and similarity score.
        """
        k = n_results or self.config.rag_top_k
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(k, count)

        query_params: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": actual_k,
        }
        if filter_metadata:
            query_params["where"] = filter_metadata

        results = self.collection.query(**query_params)

        retrieved_docs: List[Dict[str, Any]] = []
        if not results or not results.get("documents") or len(results["documents"]) == 0:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        ids = results["ids"][0] if results.get("ids") else [""] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

        for doc_text, meta, doc_id, dist in zip(docs, metas, ids, distances):
            # Convert cosine distance to similarity score in [0, 1]
            similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            retrieved_docs.append({
                "doc_id": doc_id,
                "title": meta.get("title", ""),
                "sop_code": meta.get("sop_code", ""),
                "fault_class": meta.get("fault_class", 0),
                "fault_name": meta.get("fault_name", ""),
                "estimated_hours": meta.get("estimated_hours", 1.0),
                "work_center": meta.get("work_center", "MECH-01"),
                "criticality": meta.get("criticality", "MEDIUM"),
                "primary_material": meta.get("primary_material", "NONE"),
                "content": doc_text,
                "similarity_score": round(similarity, 4),
                "distance": round(dist, 4),
            })

        return retrieved_docs
