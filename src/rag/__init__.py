"""
RAG (Retrieval Augmented Generation) module for Prometheus Lab.

This module provides semantic search capabilities over:
- Past experiment results
- Technique documentation
- Research papers (future)

Usage:
    from src.rag import RAGRetriever, ExperimentIndexer

    # First time: index experiments
    indexer = ExperimentIndexer()
    indexer.index_all()

    # Query similar experiments
    retriever = RAGRetriever()
    results = retriever.retrieve_similar_experiments(
        metrics={"wga": 0.52, "ood": 0.58},
        k=5
    )
"""

from .retriever import RAGRetriever
from .indexer import ExperimentIndexer

__all__ = ["RAGRetriever", "ExperimentIndexer"]
