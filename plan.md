# RAG & Knowledge Graph Integration Plan for Prometheus Lab

## User Choices
- **Priority**: RAG first, then Knowledge Graph
- **Vector Store**: Chroma (local)
- **Initial Data**: Past experiments only (70+ runs)
- **KG Backend**: Neo4j (full database)

## Executive Summary

Add RAG (Retrieval Augmented Generation) and Knowledge Graph capabilities to enhance the multi-agent fairness optimization system with:
1. **Semantic memory** of past experiments (Phase 1 - RAG with Chroma)
2. **Structured knowledge** of technique relationships (Phase 2 - Neo4j KG)
3. **Grounded reasoning** for strategy generation and evaluation

---

## Part 1: RAG (Retrieval Augmented Generation)

### 1.1 RAG Use Cases

| Use Case | Agent | Retrieval Source | Value |
|----------|-------|------------------|-------|
| Similar past experiments | Strategy Agent | Experiment history | Learn what worked before |
| Relevant research papers | Research Agent | Paper corpus | Ground suggestions in literature |
| Technique documentation | Strategy Agent | Technique KB | Accurate implementation details |
| Known failure modes | Critic Agent | Failure database | Avoid repeated mistakes |
| Scoring frameworks | Judge Agent | Evaluation criteria | Consistent assessment |

### 1.2 Data Sources to Index

**A. Internal Sources (Auto-generated)**
```
experiments/*.json          → Past experiment results & configs
prompts/*.md               → Technique descriptions
```

**B. External Sources (To Curate - Future)**
```
Fairness Papers:
- Sagawa et al. "Distributionally Robust Neural Networks" (Group DRO)
- Arjovsky et al. "Invariant Risk Minimization" (IRM)
- Sun & Saenko "Deep CORAL" (Domain Adaptation)
- Hardt et al. "Equality of Opportunity in Supervised Learning"
- Chouldechova "Fair Prediction with Disparate Impact"

Technique Documentation:
- Scikit-learn class_weight, sample_weight
- Domain generalization techniques catalog
- Healthcare ML fairness guidelines
```

### 1.3 RAG Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Documents   │───▶│  Chunking    │───▶│  Embeddings  │  │
│  │  (Papers,    │    │  (semantic   │    │  (all-MiniLM │  │
│  │   Results)   │    │   chunks)    │    │   or OpenAI) │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                 │            │
│                                                 ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Augmented   │◀───│  Re-ranking  │◀───│ Vector Store │  │
│  │   Prompt     │    │  & Filtering │    │   (Chroma)   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │  LLM Agent   │                                           │
│  │  (Groq)      │                                           │
│  └──────────────┘                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 Implementation Components

**New Files to Create:**

```
src/rag/
├── __init__.py          # RAG module init
├── embeddings.py        # Embedding model wrapper
├── vector_store.py      # Chroma integration
├── retriever.py         # Query & retrieval logic
├── indexer.py           # Document indexing pipeline
└── documents.py         # Document loaders & chunkers

data/knowledge_base/
├── papers/              # PDF papers (future)
├── techniques/          # Technique markdown files
└── index/               # Chroma persistent storage
```

**Core Classes:**

```python
# src/rag/retriever.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict
from langchain.schema import Document

class RAGRetriever:
    def __init__(self, vector_store_path: str = "./data/knowledge_base/index"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=self.embeddings
        )

    def retrieve_similar_experiments(self, metrics: dict, k: int = 5) -> List[Document]:
        """Find past experiments with similar metrics."""
        query = f"experiment with WGA={metrics['wga']:.2f} OOD={metrics['ood']:.2f}"
        return self.store.similarity_search(
            query, k=k,
            filter={"type": "experiment"}
        )

    def retrieve_papers(self, topic: str, k: int = 3) -> List[Document]:
        """Find relevant research papers."""
        return self.store.similarity_search(
            topic, k=k,
            filter={"type": "paper"}
        )

    def retrieve_technique_docs(self, technique: str) -> List[Document]:
        """Get documentation for a specific technique."""
        return self.store.similarity_search(
            technique, k=2,
            filter={"type": "technique"}
        )
```

```python
# src/rag/indexer.py
import json
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class ExperimentIndexer:
    def __init__(self, experiments_dir: str = "./experiments",
                 index_dir: str = "./data/knowledge_base/index"):
        self.experiments_dir = Path(experiments_dir)
        self.index_dir = index_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def load_experiments(self) -> List[Document]:
        """Load all experiment JSON files as documents."""
        documents = []
        for path in self.experiments_dir.glob("run_*.json"):
            with open(path) as f:
                data = json.load(f)

            # Create searchable text from experiment
            text = self._format_experiment(data)

            # Create document with metadata
            doc = Document(
                page_content=text,
                metadata={
                    "type": "experiment",
                    "name": data["config"]["name"],
                    "wga": data["ood"].get("worst_group_accuracy", 0),
                    "ood_acc": data["ood"]["accuracy"],
                    "id_acc": data["id"]["accuracy"],
                    "path": str(path)
                }
            )
            documents.append(doc)
        return documents

    def _format_experiment(self, data: dict) -> str:
        """Format experiment data as searchable text."""
        config = data["config"]
        ood = data["ood"]
        id_metrics = data["id"]

        return f"""
        Experiment: {config['name']}
        Strategy Configuration:
        - Class Weight: {config.get('class_weight', 'None')}
        - L2 Regularization (C): {config.get('l2_C', 1.0)}
        - Sample Fraction: {config.get('sample_frac', 1.0)}
        - Undersample Majority: {config.get('undersample_majority', False)}
        - Regularization Strength: {config.get('reg_strength', 'normal')}
        - Use Group DRO: {config.get('use_group_dro', False)}

        Results:
        - ID Accuracy: {id_metrics['accuracy']:.3f}
        - OOD Accuracy: {ood['accuracy']:.3f}
        - Worst Group Accuracy: {ood.get('worst_group_accuracy', 'N/A')}
        - ID-OOD Gap: {abs(id_metrics['accuracy'] - ood['accuracy']):.3f}
        """

    def index_all(self):
        """Index all experiments into Chroma."""
        documents = self.load_experiments()
        print(f"Indexing {len(documents)} experiments...")

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.index_dir
        )
        vectorstore.persist()
        print(f"Indexed {len(documents)} experiments to {self.index_dir}")
        return vectorstore
```

### 1.5 Agent Integration Points

**Strategy Agent Enhancement:**
```python
# src/agent_graph.py - strategy_node modification
def strategy_node(state: GraphState) -> GraphState:
    from src.rag import RAGRetriever
    retriever = RAGRetriever("./data/knowledge_base/index")

    # 1. Retrieve similar past experiments
    current_metrics = extract_metrics(state["best_run"])
    similar_experiments = retriever.retrieve_similar_experiments(current_metrics, k=3)

    # 2. Retrieve relevant technique papers
    gap = current_metrics["id_acc"] - current_metrics["ood_acc"]
    if gap > 0.1:
        papers = retriever.retrieve_papers("domain shift robust training", k=2)
    else:
        papers = retriever.retrieve_papers("worst group accuracy fairness", k=2)

    # 3. Augment prompt with retrieved context
    augmented_prompt = build_augmented_prompt(
        base_prompt=_load_prompt(),
        similar_experiments=similar_experiments,
        relevant_papers=papers
    )

    strategies, rationale = call_llm_with_prompt(augmented_prompt)
    return {**state, "proposed_configs": strategies, "strategy_rationale": rationale}
```

**Research Agent Enhancement:**
```python
# src/agent_graph.py - research_node modification
def research_node(state: GraphState) -> GraphState:
    from src.rag import RAGRetriever
    retriever = RAGRetriever("./data/knowledge_base/index")
    best = state.get("best_run")

    # Retrieve papers relevant to current weaknesses
    if best["ood"]["worst_group_accuracy"] < 0.6:
        papers = retriever.retrieve_papers("group fairness worst group performance", k=3)
    else:
        papers = retriever.retrieve_papers("domain generalization robustness", k=3)

    # Build research prompt with paper context
    research_prompt = f"""
    Analyze this experimental result:
    - OOD Accuracy: {best['ood']['accuracy']:.3f}
    - Worst Group Accuracy: {best['ood']['worst_group_accuracy']:.3f}

    Relevant research findings:
    {format_papers(papers)}

    Based on this research, what improvements would you suggest?
    """

    response = RESEARCH_LLM.invoke(research_prompt).content
    return {**state, "research_notes": response}
```

---

## Part 2: Knowledge Graph (Neo4j)

### 2.1 Knowledge Graph Use Cases

| Use Case | Component | Query Type | Value |
|----------|-----------|------------|-------|
| Technique compatibility | Strategy Agent | Path query | Avoid conflicting techniques |
| Strategy lineage | Experiment tracking | Ancestry query | Track evolution |
| Metric correlations | Judge Agent | Relationship query | Informed scoring |
| Dataset similarity | Auto-analyze | Similarity query | Transfer learning |
| Causal fairness | Analysis | Causal query | Root cause analysis |

### 2.2 Knowledge Graph Schema

```
NODES:
═══════════════════════════════════════════════════════════════
│ Type       │ Properties                                      │
├────────────┼─────────────────────────────────────────────────┤
│ Strategy   │ name, description, hyperparameters, source      │
│ Technique  │ name, category, domain_type, paper_ref          │
│ Metric     │ name, range, higher_is_better, importance       │
│ Dataset    │ name, task, n_samples, sensitive_attrs          │
│ Domain     │ name, shift_type, characteristics               │
│ Experiment │ id, timestamp, config, metrics                  │
│ Paper      │ title, authors, year, url, citations            │
│ Issue      │ name, description, severity                     │
═══════════════════════════════════════════════════════════════

RELATIONSHIPS:
═══════════════════════════════════════════════════════════════
│ Relationship        │ From       │ To         │ Properties  │
├─────────────────────┼────────────┼────────────┼─────────────┤
│ USES                │ Strategy   │ Technique  │ weight      │
│ IMPROVES            │ Strategy   │ Metric     │ delta       │
│ CONFLICTS_WITH      │ Technique  │ Technique  │ severity    │
│ COMPATIBLE_WITH     │ Technique  │ Technique  │ synergy     │
│ HANDLES             │ Technique  │ Domain     │ effectiveness│
│ CORRELATES_WITH     │ Metric     │ Metric     │ correlation │
│ PRODUCED            │ Experiment │ Metric     │ value       │
│ APPLIED_TO          │ Strategy   │ Dataset    │ effectiveness│
│ SIMILAR_TO          │ Dataset    │ Dataset    │ similarity  │
│ DERIVED_FROM        │ Strategy   │ Strategy   │ modification│
│ PROPOSES            │ Paper      │ Technique  │ year        │
│ HAS_RISK            │ Strategy   │ Issue      │ probability │
═══════════════════════════════════════════════════════════════
```

### 2.3 Knowledge Graph Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Knowledge Graph System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   Neo4j Graph Database                │  │
│  │                                                       │  │
│  │   ┌─────────┐      USES       ┌───────────┐         │  │
│  │   │Strategy │───────────────▶│ Technique │         │  │
│  │   └────┬────┘                 └─────┬─────┘         │  │
│  │        │                            │                │  │
│  │   IMPROVES                   HANDLES                │  │
│  │        │                            │                │  │
│  │        ▼                            ▼                │  │
│  │   ┌─────────┐              ┌───────────┐           │  │
│  │   │ Metric  │◀─CORRELATES─▶│  Domain   │           │  │
│  │   └─────────┘              └───────────┘           │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│                            ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Query Interface (Cypher)                 │  │
│  │                                                       │  │
│  │  MATCH (s:Strategy)-[:USES]->(t:Technique)           │  │
│  │  WHERE t.name = 'GroupDRO'                           │  │
│  │  RETURN s.name, s.effectiveness                      │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│                            ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    Agent Integration                  │  │
│  │                                                       │  │
│  │  Strategy ←── KG insights on technique compatibility  │  │
│  │  Critic   ←── KG insights on known risks             │  │
│  │  Judge    ←── KG insights on metric relationships    │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Implementation Components

**New Files to Create:**

```
src/knowledge_graph/
├── __init__.py              # KG module init
├── graph.py                 # Core FairnessKnowledgeGraph class
├── schema.py                # Node/Edge type definitions
├── queries.py               # Pre-built query templates
├── base_knowledge.py        # Curated fairness knowledge
└── persistence.py           # Save/load utilities
```

**Core Classes (Neo4j Implementation):**

```python
# src/knowledge_graph/graph.py
from typing import Dict, List, Any
from neo4j import GraphDatabase
import os

class FairnessKnowledgeGraph:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._ensure_schema()

    def _ensure_schema(self):
        """Create indexes and constraints for performance."""
        with self.driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Technique) ON (t.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Metric) ON (m.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Experiment) ON (e.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (s:Strategy) ON (s.name)")

    def load_base_knowledge(self):
        """Load curated fairness/ML knowledge into Neo4j."""
        with self.driver.session() as session:
            # Create Technique nodes
            techniques = [
                ("group_dro", "Group DRO", "robust_training",
                 "Minimizes worst-group loss", "Sagawa et al. 2020"),
                ("class_balancing", "Class Balancing", "reweighting",
                 "Balances class weights inversely to frequency", None),
                ("undersampling", "Undersampling", "reweighting",
                 "Reduces majority class samples", None),
                ("importance_sampling", "Importance Sampling", "reweighting",
                 "Reweights samples by importance", None),
                ("l2_regularization", "L2 Regularization", "regularization",
                 "Adds L2 penalty to prevent overfitting", None),
                ("domain_invariant", "Domain Invariant Features", "domain_generalization",
                 "Learns features invariant across domains", None),
            ]

            for tech_id, name, category, desc, paper in techniques:
                session.run("""
                    MERGE (t:Technique {id: $id})
                    SET t.name = $name, t.category = $category,
                        t.description = $desc, t.paper = $paper
                """, id=tech_id, name=name, category=category, desc=desc, paper=paper)

            # Create Metric nodes
            metrics = [
                ("wga", "Worst Group Accuracy", True, 1.0),
                ("ood_accuracy", "OOD Accuracy", True, 0.8),
                ("id_accuracy", "ID Accuracy", True, 0.5),
                ("fairness_gap", "Fairness Gap", False, 0.7),
            ]

            for metric_id, name, higher_better, importance in metrics:
                session.run("""
                    MERGE (m:Metric {id: $id})
                    SET m.name = $name, m.higher_is_better = $higher,
                        m.importance = $importance
                """, id=metric_id, name=name, higher=higher_better, importance=importance)

            # Create relationships
            relationships = [
                ("group_dro", "class_balancing", "COMPATIBLE_WITH", {"synergy": 0.7}),
                ("group_dro", "l2_regularization", "CONFLICTS_WITH", {"severity": 0.4}),
                ("undersampling", "class_balancing", "COMPATIBLE_WITH", {"synergy": 0.5}),
                ("group_dro", "wga", "IMPROVES", {"effectiveness": 0.8}),
                ("class_balancing", "wga", "IMPROVES", {"effectiveness": 0.6}),
            ]

            for source, target, rel_type, props in relationships:
                session.run(f"""
                    MATCH (a {{id: $source}}), (b {{id: $target}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r += $props
                """, source=source, target=target, props=props)

    def query_compatible_techniques(self, technique: str) -> List[Dict]:
        """Find techniques compatible with given technique."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t1:Technique {id: $tech})-[r:COMPATIBLE_WITH]->(t2:Technique)
                RETURN t2.id AS technique, t2.name AS name,
                       r.synergy AS synergy, r.reason AS reason
                ORDER BY r.synergy DESC
            """, tech=technique)
            return [dict(record) for record in result]

    def query_conflicting_techniques(self, technique: str) -> List[Dict]:
        """Find techniques that conflict with given technique."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t1:Technique {id: $tech})-[r:CONFLICTS_WITH]->(t2:Technique)
                RETURN t2.id AS technique, t2.name AS name,
                       r.severity AS severity, r.reason AS reason
                ORDER BY r.severity DESC
            """, tech=technique)
            return [dict(record) for record in result]

    def query_techniques_for_metric(self, metric: str, threshold: float = 0.5) -> List[Dict]:
        """Find techniques that improve a specific metric."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Technique)-[r:IMPROVES]->(m:Metric {id: $metric})
                WHERE r.effectiveness >= $threshold
                RETURN t.id AS technique, t.name AS name,
                       r.effectiveness AS effectiveness
                ORDER BY r.effectiveness DESC
            """, metric=metric, threshold=threshold)
            return [dict(record) for record in result]

    def query_strategy_risks(self, strategy_config: Dict) -> List[Dict]:
        """Identify potential risks for a strategy configuration."""
        techniques = self._extract_techniques(strategy_config)
        risks = []

        with self.driver.session() as session:
            # Check for conflicts between techniques used
            if len(techniques) > 1:
                result = session.run("""
                    MATCH (t1:Technique)-[r:CONFLICTS_WITH]->(t2:Technique)
                    WHERE t1.id IN $techs AND t2.id IN $techs
                    RETURN t1.name AS tech1, t2.name AS tech2,
                           r.severity AS severity, r.reason AS reason
                """, techs=techniques)
                risks.extend([dict(r) for r in result])

            # Check for known issues
            result = session.run("""
                MATCH (t:Technique)-[r:HAS_RISK]->(i:Issue)
                WHERE t.id IN $techs
                RETURN t.name AS technique, i.description AS issue,
                       r.probability AS probability
            """, techs=techniques)
            risks.extend([dict(r) for r in result])

        return risks

    def _extract_techniques(self, config: Dict) -> List[str]:
        """Extract technique IDs from a strategy config."""
        techniques = []
        if config.get("use_group_dro"):
            techniques.append("group_dro")
        if config.get("class_weight") == "balanced":
            techniques.append("class_balancing")
        if config.get("undersample_majority"):
            techniques.append("undersampling")
        if config.get("l2_C", 1.0) != 1.0:
            techniques.append("l2_regularization")
        return techniques

    def add_experiment_outcome(self, config: Dict, metrics: Dict):
        """Learn from experiment results - add to graph."""
        exp_id = f"exp_{config['name']}_{hash(str(config)) % 10000}"

        with self.driver.session() as session:
            # Create experiment node
            session.run("""
                MERGE (e:Experiment {id: $exp_id})
                SET e.config = $config,
                    e.wga = $wga,
                    e.ood_acc = $ood_acc,
                    e.timestamp = datetime()
            """, exp_id=exp_id, config=str(config),
                 wga=metrics.get("worst_group_accuracy", 0),
                 ood_acc=metrics.get("ood_accuracy", 0))

            # Link to techniques used
            for tech in self._extract_techniques(config):
                session.run("""
                    MATCH (e:Experiment {id: $exp_id}), (t:Technique {id: $tech})
                    MERGE (e)-[:USED]->(t)
                """, exp_id=exp_id, tech=tech)

    def close(self):
        """Close the database connection."""
        self.driver.close()
```

### 2.5 Agent Integration Points

**Strategy Agent Enhancement:**
```python
# src/agent_graph.py - strategy_node with KG
def strategy_node(state: GraphState) -> GraphState:
    from src.knowledge_graph import FairnessKnowledgeGraph
    kg = FairnessKnowledgeGraph()

    try:
        # 1. Query techniques that improve WGA
        wga_techniques = kg.query_techniques_for_metric("wga", threshold=0.5)

        # 2. Check compatibility between potential techniques
        compatible_combos = []
        for t1 in wga_techniques:
            compatible = kg.query_compatible_techniques(t1["technique"])
            for t2 in compatible:
                compatible_combos.append({
                    "tech1": t1["technique"],
                    "tech2": t2["technique"],
                    "synergy": t2["synergy"]
                })

        # 3. Build augmented prompt with KG insights
        kg_context = f"""
        Techniques known to improve Worst Group Accuracy:
        {format_techniques(wga_techniques)}

        Recommended technique combinations (high synergy):
        {format_combos(sorted(compatible_combos, key=lambda x: -x['synergy'])[:3])}

        Avoid combining techniques that conflict.
        """

        augmented_prompt = _load_prompt() + "\n\n" + kg_context
        strategies, rationale = call_llm_with_prompt(augmented_prompt)
    finally:
        kg.close()

    return {**state, "proposed_configs": strategies, "strategy_rationale": rationale}
```

**Critic Agent Enhancement:**
```python
# src/agent_graph.py - critic_node with KG
def critic_node(state: GraphState) -> GraphState:
    from src.knowledge_graph import FairnessKnowledgeGraph
    kg = FairnessKnowledgeGraph()
    cfgs = state.get("proposed_configs", [])

    try:
        critiques = []
        for cfg in cfgs:
            # Query KG for risks
            risks = kg.query_strategy_risks(cfg)
            critiques.append({
                "strategy": cfg["name"],
                "risks": risks
            })

        # Build critic prompt with KG-informed risks
        critic_prompt = f"""
        Evaluate these proposed strategies:
        {json.dumps(cfgs, indent=2)}

        Known risks from knowledge graph:
        {format_critiques(critiques)}

        Provide your critique focusing on feasibility and potential issues.
        """

        response = CRITIC_LLM.invoke(critic_prompt).content
    finally:
        kg.close()

    return {**state, "critic_notes": response}
```

---

## Part 3: Combined RAG + KG Architecture

### 3.1 Hybrid Retrieval System

```
┌─────────────────────────────────────────────────────────────────┐
│                   Hybrid RAG + KG System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query: "How to improve WGA from 0.52 with 15% domain gap?"     │
│                           │                                      │
│              ┌────────────┴────────────┐                        │
│              ▼                         ▼                         │
│  ┌───────────────────┐    ┌───────────────────┐                │
│  │   RAG Retrieval   │    │   KG Traversal    │                │
│  │                   │    │                   │                │
│  │ • Similar exps    │    │ • Technique rels  │                │
│  │ • Relevant papers │    │ • Metric deps     │                │
│  │ • Technique docs  │    │ • Risk paths      │                │
│  └─────────┬─────────┘    └─────────┬─────────┘                │
│            │                        │                           │
│            └───────────┬────────────┘                           │
│                        ▼                                         │
│            ┌───────────────────┐                                │
│            │   Context Fusion  │                                │
│            │                   │                                │
│            │ Combine semantic  │                                │
│            │ + structured      │                                │
│            │ knowledge         │                                │
│            └─────────┬─────────┘                                │
│                      ▼                                           │
│            ┌───────────────────┐                                │
│            │  Augmented Prompt │                                │
│            │                   │                                │
│            │ Base prompt +     │                                │
│            │ RAG context +     │                                │
│            │ KG insights       │                                │
│            └─────────┬─────────┘                                │
│                      ▼                                           │
│            ┌───────────────────┐                                │
│            │    LLM Agent      │                                │
│            └───────────────────┘                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Implementation Phases

### Phase 1: RAG Foundation
- [ ] Install dependencies: `chromadb`, `sentence-transformers`, `langchain`
- [ ] Create `src/rag/` module structure
- [ ] Implement `ExperimentIndexer` class
- [ ] Implement `RAGRetriever` class
- [ ] Index all 70+ existing experiments from `experiments/`
- [ ] Integrate RAG into `strategy_node`
- [ ] Integrate RAG into `research_node`
- [ ] Test retrieval quality

### Phase 2: Neo4j Knowledge Graph
- [ ] Install Neo4j (Docker: `docker run -p 7474:7474 -p 7687:7687 neo4j`)
- [ ] Install dependency: `neo4j` Python driver
- [ ] Create `src/knowledge_graph/` module structure
- [ ] Implement `FairnessKnowledgeGraph` class
- [ ] Design and load base fairness knowledge (techniques, metrics, relationships)
- [ ] Integrate KG into `critic_node` for risk analysis
- [ ] Add experiment outcome learning to KG
- [ ] Test Cypher queries

### Phase 3: Enhanced Integration
- [ ] Combine RAG + KG for hybrid queries
- [ ] Add technique documentation to RAG corpus
- [ ] Create unified query interface for agents
- [ ] Add auto-indexing of new experiments after each run

### Phase 4: Advanced Features (Future)
- [ ] Index academic papers (PDF processing with `pypdf`)
- [ ] Web UI for KG exploration
- [ ] Causal reasoning integration
- [ ] Cross-dataset transfer learning

---

## Part 5: Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/rag/__init__.py` | RAG module init, exports |
| `src/rag/embeddings.py` | Embedding model wrapper |
| `src/rag/vector_store.py` | Chroma integration |
| `src/rag/retriever.py` | RAGRetriever class |
| `src/rag/indexer.py` | ExperimentIndexer class |
| `src/knowledge_graph/__init__.py` | KG module init, exports |
| `src/knowledge_graph/graph.py` | FairnessKnowledgeGraph class |
| `src/knowledge_graph/schema.py` | Node/Edge type definitions |
| `src/knowledge_graph/queries.py` | Query templates |
| `src/knowledge_graph/base_knowledge.py` | Curated fairness knowledge |
| `data/knowledge_base/index/` | Chroma persistent storage |
| `data/knowledge_base/techniques/` | Technique documentation |

### Files to Modify

| File | Changes |
|------|---------|
| `src/agent_graph.py` | Add RAG/KG to strategy_node, research_node, critic_node |
| `src/auto_analyze.py` | Add KG-based dataset recommendations |
| `src/llm_client.py` | Add context fusion helpers |
| `src/run_experiment.py` | Add KG outcome learning after experiments |
| `requirements.txt` | Add new dependencies |

---

## Part 6: Dependencies to Add

```txt
# requirements.txt additions

# Phase 1: RAG
chromadb>=0.4.0              # Vector store (Chroma)
sentence-transformers>=2.2.0 # Local embeddings (all-MiniLM-L6-v2)
langchain>=0.1.0             # RAG utilities & document loaders
langchain-community>=0.0.10  # Community integrations

# Phase 2: Knowledge Graph
neo4j>=5.0.0                 # Neo4j Python driver

# Future: Paper Processing (Phase 4)
pypdf>=3.0.0                 # PDF parsing for papers
```

---

## Part 7: Example Usage

### RAG Query Example
```python
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
for doc in results:
    print(f"Strategy: {doc.metadata['name']}")
    print(f"WGA: {doc.metadata['wga']}, OOD: {doc.metadata['ood_acc']}")
```

### KG Query Example
```python
from src.knowledge_graph import FairnessKnowledgeGraph

kg = FairnessKnowledgeGraph()

# Load base knowledge (first time)
kg.load_base_knowledge()

# Find techniques for improving WGA
techniques = kg.query_techniques_for_metric("wga")
print("Techniques for WGA:", techniques)

# Check compatibility
compatible = kg.query_compatible_techniques("group_dro")
print("Compatible with Group DRO:", compatible)

# Get strategy risks
risks = kg.query_strategy_risks({
    "name": "aggressive_dro",
    "use_group_dro": True,
    "l2_C": 0.1,
    "reg_strength": "strong"
})
print("Risks:", risks)

kg.close()
```

---

## Summary

| Feature | Benefit | Phase |
|---------|---------|-------|
| **RAG for Experiments** | Learn from 70+ past runs | Phase 1 |
| **Chroma Vector Store** | Fast local semantic search | Phase 1 |
| **Neo4j Knowledge Graph** | Structured technique relationships | Phase 2 |
| **KG Risk Analysis** | Avoid conflicting techniques | Phase 2 |
| **Hybrid RAG+KG** | Best of both worlds | Phase 3 |
| **Auto-learning** | Improve over time | Phase 3 |

This plan provides a comprehensive roadmap for adding intelligent memory and reasoning to Prometheus Lab through RAG and Knowledge Graph integration.
