"""
RAG Retriever for semantic search over experiments and documents.

Provides retrieval capabilities for:
- Similar past experiments
- Technique documentation
- Research papers (future)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from langchain.schema import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Document = None


class RAGRetriever:
    """Retriever for semantic search over indexed documents."""

    def __init__(
        self,
        index_dir: str = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the RAG retriever.

        Args:
            index_dir: Path to Chroma index directory
            embedding_model: HuggingFace model name for embeddings
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "RAG dependencies not installed. Run: "
                "pip install chromadb sentence-transformers langchain langchain-community"
            )

        # Resolve path relative to project root
        project_root = Path(__file__).resolve().parents[2]
        self.index_dir = str(Path(index_dir) if index_dir else project_root / "data" / "knowledge_base" / "index")

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Load vector store
        self.store = Chroma(
            persist_directory=self.index_dir,
            embedding_function=self.embeddings
        )

    def retrieve_similar_experiments(
        self,
        metrics: Dict[str, float],
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Find past experiments with similar metrics.

        Args:
            metrics: Dict with keys like 'wga', 'ood', 'gap'
            k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of similar experiment documents
        """
        # Build semantic query from metrics
        wga = metrics.get("wga", metrics.get("worst_group_accuracy", 0))
        ood = metrics.get("ood", metrics.get("ood_acc", metrics.get("ood_accuracy", 0)))
        gap = metrics.get("gap", 0)

        query = f"""
        Find experiments with similar performance:
        - Worst Group Accuracy around {wga:.2f}
        - OOD Accuracy around {ood:.2f}
        - ID-OOD Gap around {gap:.2f}
        Looking for strategies that achieved these metrics or improved upon them.
        """

        # Apply filters
        search_filter = {"type": "experiment"}
        if filter_dict:
            search_filter.update(filter_dict)

        try:
            results = self.store.similarity_search(
                query,
                k=k,
                filter=search_filter
            )
            return results
        except Exception as e:
            print(f"Error retrieving experiments: {e}")
            return []

    def retrieve_by_query(
        self,
        query: str,
        k: int = 5,
        doc_type: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve documents by free-text query.

        Args:
            query: Natural language query
            k: Number of results to return
            doc_type: Filter by document type ('experiment', 'technique', 'paper')

        Returns:
            List of matching documents
        """
        search_filter = {}
        if doc_type:
            search_filter["type"] = doc_type

        try:
            if search_filter:
                results = self.store.similarity_search(query, k=k, filter=search_filter)
            else:
                results = self.store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def retrieve_papers(self, topic: str, k: int = 3) -> List[Document]:
        """
        Find relevant research papers.

        Args:
            topic: Research topic to search for
            k: Number of results to return

        Returns:
            List of paper documents
        """
        return self.retrieve_by_query(topic, k=k, doc_type="paper")

    def retrieve_technique_docs(self, technique: str, k: int = 2) -> List[Document]:
        """
        Get documentation for a specific technique.

        Args:
            technique: Technique name or description
            k: Number of results to return

        Returns:
            List of technique documentation documents
        """
        return self.retrieve_by_query(technique, k=k, doc_type="technique")

    def retrieve_experiments_for_strategy(
        self,
        strategy_description: str,
        k: int = 5
    ) -> List[Document]:
        """
        Find experiments that used a similar strategy.

        Args:
            strategy_description: Description of the strategy to search for
            k: Number of results to return

        Returns:
            List of experiment documents
        """
        query = f"Experiments using strategy: {strategy_description}"
        return self.retrieve_by_query(query, k=k, doc_type="experiment")

    def retrieve_high_performing_experiments(
        self,
        metric: str = "wga",
        threshold: float = 0.6,
        k: int = 5
    ) -> List[Document]:
        """
        Find experiments with high performance on a specific metric.

        Args:
            metric: Metric to filter by ('wga', 'ood_acc', etc.)
            threshold: Minimum value for the metric
            k: Number of results to return

        Returns:
            List of high-performing experiment documents
        """
        query = f"High performing experiments with {metric} above {threshold}"

        try:
            # Chroma filter for numeric comparison
            results = self.store.similarity_search(
                query,
                k=k * 2,  # Get more results to filter
                filter={"type": "experiment"}
            )

            # Post-filter by metric threshold
            filtered = []
            for doc in results:
                if doc.metadata.get(metric, 0) >= threshold:
                    filtered.append(doc)
                if len(filtered) >= k:
                    break

            return filtered
        except Exception as e:
            print(f"Error retrieving high-performing experiments: {e}")
            return []

    def get_experiment_insights(self, metrics: Dict[str, float]) -> str:
        """
        Generate insights from similar past experiments.

        Args:
            metrics: Current experiment metrics

        Returns:
            Formatted string with insights
        """
        similar = self.retrieve_similar_experiments(metrics, k=3)

        if not similar:
            return "No similar experiments found in the database."

        insights = "## Similar Past Experiments\n\n"

        for i, doc in enumerate(similar, 1):
            meta = doc.metadata
            insights += f"### Experiment {i}: {meta.get('name', 'Unknown')}\n"
            insights += f"- WGA: {meta.get('wga', 'N/A'):.3f}\n"
            insights += f"- OOD Accuracy: {meta.get('ood_acc', 'N/A'):.3f}\n"
            insights += f"- ID-OOD Gap: {meta.get('gap', 'N/A'):.3f}\n"
            insights += f"- Group DRO: {meta.get('use_group_dro', False)}\n"
            insights += f"- Class Weight: {meta.get('class_weight', 'None')}\n\n"

        return insights

    def format_retrieved_context(
        self,
        experiments: List[Document],
        papers: Optional[List[Document]] = None,
        max_length: int = 2000
    ) -> str:
        """
        Format retrieved documents as context for LLM prompts.

        Args:
            experiments: Retrieved experiment documents
            papers: Retrieved paper documents (optional)
            max_length: Maximum context length

        Returns:
            Formatted context string
        """
        context = ""

        # Add experiment context
        if experiments:
            context += "## Relevant Past Experiments\n\n"
            for i, doc in enumerate(experiments[:3], 1):
                meta = doc.metadata
                context += f"**Experiment {i}: {meta.get('name', 'Unknown')}**\n"
                context += f"- WGA: {meta.get('wga', 0):.3f}, OOD: {meta.get('ood_acc', 0):.3f}\n"
                context += f"- Strategy: Group DRO={meta.get('use_group_dro', False)}, "
                context += f"Class Weight={meta.get('class_weight', 'None')}\n\n"

        # Add paper context
        if papers:
            context += "## Relevant Research\n\n"
            for i, doc in enumerate(papers[:2], 1):
                # Truncate content if needed
                content = doc.page_content[:500]
                context += f"**Source {i}:**\n{content}...\n\n"

        # Truncate if too long
        if len(context) > max_length:
            context = context[:max_length] + "\n[Context truncated...]"

        return context


def test_retriever():
    """CLI function to test the retriever."""
    print("=" * 50)
    print("Prometheus Lab - RAG Retriever Test")
    print("=" * 50)

    try:
        retriever = RAGRetriever()

        # Test retrieval
        print("\nTesting retrieval with sample metrics...")
        results = retriever.retrieve_similar_experiments(
            metrics={"wga": 0.55, "ood": 0.58, "gap": 0.10},
            k=3
        )

        print(f"\nFound {len(results)} similar experiments:")
        for doc in results:
            print(f"  - {doc.metadata.get('name')}: WGA={doc.metadata.get('wga', 0):.3f}")

        # Test insights generation
        print("\nGenerating insights...")
        insights = retriever.get_experiment_insights(
            {"wga": 0.55, "ood": 0.58, "gap": 0.10}
        )
        print(insights)

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run the indexer first: python -m src.rag.indexer")


if __name__ == "__main__":
    test_retriever()
