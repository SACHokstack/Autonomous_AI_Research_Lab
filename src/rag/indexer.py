"""
Experiment Indexer for RAG system.

Indexes experiment JSON files into Chroma vector store for semantic search.
"""

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

try:
    from langchain.schema import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Document = None


class ExperimentIndexer:
    """Indexes experiment results into a Chroma vector store."""

    def __init__(
        self,
        experiments_dir: str = None,
        index_dir: str = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the experiment indexer.

        Args:
            experiments_dir: Path to experiments directory (default: ./experiments)
            index_dir: Path to Chroma index directory (default: ./data/knowledge_base/index)
            embedding_model: HuggingFace model name for embeddings
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "RAG dependencies not installed. Run: "
                "pip install chromadb sentence-transformers langchain langchain-community"
            )

        # Resolve paths relative to project root
        project_root = Path(__file__).resolve().parents[2]
        self.experiments_dir = Path(experiments_dir) if experiments_dir else project_root / "experiments"
        self.index_dir = str(Path(index_dir) if index_dir else project_root / "data" / "knowledge_base" / "index")

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def load_experiments(self) -> List[Document]:
        """
        Load all experiment JSON files as LangChain Documents.

        Returns:
            List of Document objects with experiment data
        """
        documents = []

        for path in self.experiments_dir.glob("run_*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)

                # Create searchable text from experiment
                text = self._format_experiment(data)

                # Extract metrics for metadata
                config = data.get("config", {})
                ood = data.get("ood", {})
                id_metrics = data.get("id", {})

                # Calculate gap
                id_acc = id_metrics.get("accuracy", 0)
                ood_acc = ood.get("accuracy", 0)
                gap = abs(id_acc - ood_acc)

                # Create document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "type": "experiment",
                        "name": config.get("name", "unknown"),
                        "wga": ood.get("worst_group_accuracy", 0),
                        "ood_acc": ood_acc,
                        "id_acc": id_acc,
                        "gap": gap,
                        "class_weight": config.get("class_weight"),
                        "l2_C": config.get("l2_C", 1.0),
                        "use_group_dro": config.get("use_group_dro", False),
                        "undersample_majority": config.get("undersample_majority", False),
                        "path": str(path)
                    }
                )
                documents.append(doc)
                print(f"  Loaded: {path.name}")

            except Exception as e:
                print(f"  Error loading {path}: {e}")
                continue

        return documents

    def _format_experiment(self, data: dict) -> str:
        """
        Format experiment data as searchable text.

        Args:
            data: Experiment JSON data

        Returns:
            Formatted text for embedding
        """
        config = data.get("config", {})
        ood = data.get("ood", {})
        id_metrics = data.get("id", {})

        # Build text representation
        text = f"""
Experiment: {config.get('name', 'unknown')}

Strategy Configuration:
- Class Weight: {config.get('class_weight', 'None')}
- L2 Regularization (C): {config.get('l2_C', 1.0)}
- Sample Fraction: {config.get('sample_frac', 1.0)}
- Undersample Majority: {config.get('undersample_majority', False)}
- Regularization Strength: {config.get('reg_strength', 'normal')}
- Use Group DRO: {config.get('use_group_dro', False)}

Performance Results:
- ID Accuracy: {id_metrics.get('accuracy', 'N/A'):.3f if isinstance(id_metrics.get('accuracy'), (int, float)) else 'N/A'}
- ID AUC: {id_metrics.get('auc', 'N/A'):.3f if isinstance(id_metrics.get('auc'), (int, float)) else 'N/A'}
- OOD Accuracy: {ood.get('accuracy', 'N/A'):.3f if isinstance(ood.get('accuracy'), (int, float)) else 'N/A'}
- OOD AUC: {ood.get('auc', 'N/A'):.3f if isinstance(ood.get('auc'), (int, float)) else 'N/A'}
- Worst Group Accuracy: {ood.get('worst_group_accuracy', 'N/A'):.3f if isinstance(ood.get('worst_group_accuracy'), (int, float)) else 'N/A'}
- ID-OOD Gap: {abs(id_metrics.get('accuracy', 0) - ood.get('accuracy', 0)):.3f}
"""

        # Add group accuracies if available
        group_acc = ood.get("group_accuracy", {})
        if group_acc:
            text += "\nGroup-wise Accuracies:\n"
            for group, acc in group_acc.items():
                text += f"- {group}: {acc:.3f}\n"

        return text.strip()

    def index_all(self, force_reindex: bool = False) -> Optional[Chroma]:
        """
        Index all experiments into Chroma vector store.

        Args:
            force_reindex: If True, recreate index even if it exists

        Returns:
            Chroma vector store instance
        """
        index_path = Path(self.index_dir)

        # Check if index already exists
        if index_path.exists() and not force_reindex:
            chroma_files = list(index_path.glob("*.bin")) + list(index_path.glob("*.pkl"))
            if chroma_files:
                print(f"Index already exists at {self.index_dir}")
                print("Use force_reindex=True to recreate")
                return Chroma(
                    persist_directory=self.index_dir,
                    embedding_function=self.embeddings
                )

        # Load experiments
        print(f"Loading experiments from {self.experiments_dir}...")
        documents = self.load_experiments()

        if not documents:
            print("No experiments found to index!")
            return None

        print(f"\nIndexing {len(documents)} experiments...")

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.index_dir
        )

        print(f"Successfully indexed {len(documents)} experiments to {self.index_dir}")
        return vectorstore

    def add_experiment(self, experiment_path: str) -> bool:
        """
        Add a single new experiment to the existing index.

        Args:
            experiment_path: Path to experiment JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(experiment_path)
            with open(path) as f:
                data = json.load(f)

            text = self._format_experiment(data)
            config = data.get("config", {})
            ood = data.get("ood", {})
            id_metrics = data.get("id", {})

            doc = Document(
                page_content=text,
                metadata={
                    "type": "experiment",
                    "name": config.get("name", "unknown"),
                    "wga": ood.get("worst_group_accuracy", 0),
                    "ood_acc": ood.get("accuracy", 0),
                    "id_acc": id_metrics.get("accuracy", 0),
                    "gap": abs(id_metrics.get("accuracy", 0) - ood.get("accuracy", 0)),
                    "path": str(path)
                }
            )

            # Load existing store and add document
            vectorstore = Chroma(
                persist_directory=self.index_dir,
                embedding_function=self.embeddings
            )
            vectorstore.add_documents([doc])
            print(f"Added experiment: {path.name}")
            return True

        except Exception as e:
            print(f"Error adding experiment: {e}")
            return False


def index_experiments():
    """CLI function to index all experiments."""
    print("=" * 50)
    print("Prometheus Lab - Experiment Indexer")
    print("=" * 50)

    indexer = ExperimentIndexer()
    indexer.index_all(force_reindex=True)


if __name__ == "__main__":
    index_experiments()
