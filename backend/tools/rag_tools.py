# rag_tools.py
"""
RAG Tools: Knowledge Base with Hybrid Search and Semantic Column Selection.
Consolidates 'knowledge_base.py' and 'semantic_column_selector.py'.
"""

import json
import logging
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import math

logger = logging.getLogger(__name__)

# =============================================================================
# KNOWLEDGE BASE (HYBRID SEARCH)
# =============================================================================

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class SimpleBM25:
    """
    Simple BM25 implementation for hybrid search without extra dependencies.
    """
    def __init__(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self._initialize(corpus)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                nd[word] = nd.get(word, 0) + 1

        self.avgdl = num_doc / self.corpus_size
        
        for word, freq in nd.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
            self.idf[word] = idf

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0] * self.corpus_size
        for q in query:
            q_freq = self.idf.get(q)
            if q_freq:
                for i in range(self.corpus_size):
                    doc_freq = self.doc_freqs[i].get(q, 0)
                    doc_len = self.doc_len[i]
                    score = (q_freq * doc_freq * (2.5)) / (doc_freq + 1.5 * (1 - 0.75 + 0.75 * doc_len / self.avgdl))
                    scores[i] += score
        return scores

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

class KnowledgeBase:
    """
    Hybrid Search Knowledge Base using Ollama Embeddings + BM25
    """
    def __init__(self, llm_client=None, storage_dir: str = "knowledge_base"):
        self.llm_client = llm_client
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.documents: List[Document] = []
        self.bm25 = None
        self.is_indexed = False
        
    def add_documents(self, docs: List[Dict[str, Any]]):
        """
        Add documents to the KB.
        docs: List of {"id": str, "content": str, "metadata": dict}
        """
        for d in docs:
            self.documents.append(Document(
                id=d.get("id", str(len(self.documents))),
                content=d["content"],
                metadata=d.get("metadata", {})
            ))
        self.is_indexed = False
        
    def build_index(self):
        """
        Generate embeddings and build BM25 index
        """
        if not self.documents:
            return

        logger.info(f"Building index for {len(self.documents)} documents...")
        
        # 1. BM25 Index
        corpus_tokens = [SimpleBM25.tokenize(d.content) for d in self.documents]
        self.bm25 = SimpleBM25(corpus_tokens)
        
        # 2. Vector Index (Embeddings)
        if self.llm_client:
            for doc in self.documents:
                if doc.embedding is None:
                    try:
                        # Use Ollama embedding endpoint
                        response = self.llm_client.embeddings.create(
                            model="nomic-embed-text", # or llama3.2 depending on what's pulled
                            input=doc.content
                        )
                        doc.embedding = response.data[0].embedding
                    except Exception as e:
                        logger.warning(f"Failed to embed doc {doc.id}: {e}")
                        # Fallback: random embedding or skip
                        doc.embedding = [0.0] * 768 # Placeholder
        
        self.is_indexed = True
        logger.info("Index built successfully.")

    def search(self, query: str, top_k: int = 3, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Hybrid Search: alpha * VectorScore + (1-alpha) * BM25Score
        """
        if not self.documents:
            return []
            
        if not self.is_indexed:
            self.build_index()
            
        # 1. BM25 Scores
        query_tokens = SimpleBM25.tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        if max_bm25 == 0: max_bm25 = 1.0
        bm25_norm = [s / max_bm25 for s in bm25_scores]
        
        # 2. Vector Scores
        vector_scores = [0.0] * len(self.documents)
        if self.llm_client:
            try:
                q_resp = self.llm_client.embeddings.create(
                    model="nomic-embed-text",
                    input=query
                )
                q_emb = np.array(q_resp.data[0].embedding)
                
                # Cosine similarity
                for i, doc in enumerate(self.documents):
                    if doc.embedding:
                        d_emb = np.array(doc.embedding)
                        # Cosine sim: (A . B) / (|A| |B|)
                        norm_q = np.linalg.norm(q_emb)
                        norm_d = np.linalg.norm(d_emb)
                        if norm_q > 0 and norm_d > 0:
                            vector_scores[i] = np.dot(q_emb, d_emb) / (norm_q * norm_d)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # 3. Hybrid Merge
        final_scores = []
        for i in range(len(self.documents)):
            score = (alpha * vector_scores[i]) + ((1 - alpha) * bm25_norm[i])
            final_scores.append((self.documents[i], score))
            
        # Sort
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for doc, score in final_scores[:top_k]:
            results.append({
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": float(score)
            })
            
        return results


# =============================================================================
# SEMANTIC COLUMN SELECTOR
# =============================================================================

class SemanticColumnSelector:
    """
    Selects columns based on semantic meaning using embeddings.
    Useful for "Smart Extraction" when there are 1000+ columns.
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    def find_best_columns(self, query_sensors: List[str], all_columns: List[str], threshold: float = 0.6) -> Dict[str, List[str]]:
        """
        Match requested sensor names to actual columns.
        Returns: {"requested_name": ["col1", "col2"]}
        """
        if not self.llm_client or not all_columns:
            return {q: [] for q in query_sensors}
            
        results = {}
        
        # 1. Embed all columns (Cache this in production!)
        col_embeddings = {}
        try:
            for col in all_columns:
                resp = self.llm_client.embeddings.create(
                    model="nomic-embed-text",
                    input=col
                )
                col_embeddings[col] = np.array(resp.data[0].embedding)
                
        except Exception as e:
            logger.error(f"Column embedding failed: {e}")
            # Fallback to string matching
            return self._fallback_match(query_sensors, all_columns)

        # 2. Embed queries and match
        for q in query_sensors:
            try:
                resp = self.llm_client.embeddings.create(
                    model="nomic-embed-text",
                    input=q
                )
                q_emb = np.array(resp.data[0].embedding)
                
                matches = []
                for col, col_emb in col_embeddings.items():
                    # Cosine sim
                    score = np.dot(q_emb, col_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(col_emb))
                    if score > threshold:
                        matches.append((col, score))
                
                # Sort by score
                matches.sort(key=lambda x: x[1], reverse=True)
                results[q] = [m[0] for m in matches[:3]] # Top 3 matches
                
            except Exception as e:
                logger.error(f"Query embedding failed for {q}: {e}")
                results[q] = []
                
        return results

    def _fallback_match(self, query_sensors: List[str], all_columns: List[str]) -> Dict[str, List[str]]:
        """Simple string matching fallback"""
        results = {}
        for q in query_sensors:
            q_lower = q.lower()
            matches = [c for c in all_columns if q_lower in c.lower()]
            results[q] = matches[:3]
        return results
