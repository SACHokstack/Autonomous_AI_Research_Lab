"""
Fairness Knowledge Graph using Neo4j.

Provides structured knowledge about:
- ML fairness techniques and their relationships
- Metric correlations and dependencies
- Strategy risk analysis
- Experiment outcome learning
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None


class FairnessKnowledgeGraph:
    """
    Knowledge Graph for ML Fairness using Neo4j.

    Stores and queries:
    - Techniques and their relationships
    - Metrics and their correlations
    - Experiments and outcomes
    - Strategy risks and issues
    """

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        auto_load_knowledge: bool = False
    ):
        """
        Initialize the Knowledge Graph connection.

        Args:
            uri: Neo4j connection URI (default: from NEO4J_URI env var)
            user: Neo4j username (default: from NEO4J_USER env var)
            password: Neo4j password (default: from NEO4J_PASSWORD env var)
            auto_load_knowledge: If True, load base knowledge on init
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "Neo4j driver not installed. Run: pip install neo4j"
            )

        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )

        # Verify connection
        self._verify_connection()

        # Create indexes
        self._ensure_schema()

        if auto_load_knowledge:
            self.load_base_knowledge()

    def _verify_connection(self):
        """Verify Neo4j connection is working."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j at {self.uri}: {e}")

    def _ensure_schema(self):
        """Create indexes and constraints for performance."""
        with self.driver.session() as session:
            # Create indexes for common queries
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (t:Technique) ON (t.id)",
                "CREATE INDEX IF NOT EXISTS FOR (t:Technique) ON (t.name)",
                "CREATE INDEX IF NOT EXISTS FOR (m:Metric) ON (m.id)",
                "CREATE INDEX IF NOT EXISTS FOR (m:Metric) ON (m.name)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Experiment) ON (e.id)",
                "CREATE INDEX IF NOT EXISTS FOR (s:Strategy) ON (s.name)",
                "CREATE INDEX IF NOT EXISTS FOR (i:Issue) ON (i.id)",
            ]
            for index_query in indexes:
                try:
                    session.run(index_query)
                except Exception:
                    pass  # Index might already exist

    def load_base_knowledge(self):
        """Load curated fairness/ML knowledge into Neo4j."""
        print("Loading base fairness knowledge...")

        with self.driver.session() as session:
            # Create Technique nodes
            techniques = [
                {
                    "id": "group_dro",
                    "name": "Group DRO",
                    "category": "robust_training",
                    "description": "Distributionally Robust Optimization that minimizes worst-group loss",
                    "paper": "Sagawa et al. 2020",
                    "effectiveness": 0.8
                },
                {
                    "id": "class_balancing",
                    "name": "Class Balancing",
                    "category": "reweighting",
                    "description": "Balances class weights inversely proportional to class frequency",
                    "paper": None,
                    "effectiveness": 0.6
                },
                {
                    "id": "undersampling",
                    "name": "Majority Undersampling",
                    "category": "reweighting",
                    "description": "Reduces majority class samples to balance the dataset",
                    "paper": None,
                    "effectiveness": 0.5
                },
                {
                    "id": "importance_sampling",
                    "name": "Importance Sampling",
                    "category": "reweighting",
                    "description": "Reweights samples based on importance for target distribution",
                    "paper": None,
                    "effectiveness": 0.6
                },
                {
                    "id": "l2_regularization",
                    "name": "L2 Regularization",
                    "category": "regularization",
                    "description": "Adds L2 penalty to prevent overfitting and improve generalization",
                    "paper": None,
                    "effectiveness": 0.5
                },
                {
                    "id": "strong_l2",
                    "name": "Strong L2 Regularization",
                    "category": "regularization",
                    "description": "Aggressive L2 penalty (C=0.1) for high-bias, low-variance models",
                    "paper": None,
                    "effectiveness": 0.4
                },
                {
                    "id": "domain_invariant",
                    "name": "Domain Invariant Features",
                    "category": "domain_generalization",
                    "description": "Learns features that are invariant across different domains",
                    "paper": "Ganin et al. 2016",
                    "effectiveness": 0.7
                },
                {
                    "id": "focal_loss",
                    "name": "Focal Loss",
                    "category": "robust_training",
                    "description": "Down-weights easy examples to focus on hard examples",
                    "paper": "Lin et al. 2017",
                    "effectiveness": 0.6
                },
                {
                    "id": "irm",
                    "name": "Invariant Risk Minimization",
                    "category": "domain_generalization",
                    "description": "Learns invariant predictors across environments",
                    "paper": "Arjovsky et al. 2019",
                    "effectiveness": 0.7
                },
            ]

            for tech in techniques:
                session.run("""
                    MERGE (t:Technique {id: $id})
                    SET t.name = $name,
                        t.category = $category,
                        t.description = $description,
                        t.paper = $paper,
                        t.effectiveness = $effectiveness
                """, **tech)

            # Create Metric nodes
            metrics = [
                {
                    "id": "wga",
                    "name": "Worst Group Accuracy",
                    "description": "Minimum accuracy across all demographic groups",
                    "higher_is_better": True,
                    "importance": 1.0
                },
                {
                    "id": "ood_accuracy",
                    "name": "OOD Accuracy",
                    "description": "Accuracy on out-of-distribution test set",
                    "higher_is_better": True,
                    "importance": 0.8
                },
                {
                    "id": "id_accuracy",
                    "name": "ID Accuracy",
                    "description": "Accuracy on in-distribution test set",
                    "higher_is_better": True,
                    "importance": 0.5
                },
                {
                    "id": "fairness_gap",
                    "name": "Fairness Gap",
                    "description": "Difference between best and worst group accuracy",
                    "higher_is_better": False,
                    "importance": 0.7
                },
                {
                    "id": "robustness_gap",
                    "name": "Robustness Gap",
                    "description": "Difference between ID and OOD accuracy",
                    "higher_is_better": False,
                    "importance": 0.6
                },
            ]

            for metric in metrics:
                session.run("""
                    MERGE (m:Metric {id: $id})
                    SET m.name = $name,
                        m.description = $description,
                        m.higher_is_better = $higher_is_better,
                        m.importance = $importance
                """, **metric)

            # Create Issue nodes
            issues = [
                {
                    "id": "overfitting",
                    "name": "Overfitting Risk",
                    "description": "Model may overfit to training distribution",
                    "severity": 0.7
                },
                {
                    "id": "underfitting",
                    "name": "Underfitting Risk",
                    "description": "Model may underfit due to excessive regularization",
                    "severity": 0.6
                },
                {
                    "id": "minority_neglect",
                    "name": "Minority Group Neglect",
                    "description": "Small groups may be ignored during optimization",
                    "severity": 0.8
                },
                {
                    "id": "domain_shift",
                    "name": "Domain Shift Sensitivity",
                    "description": "Performance may degrade under distribution shift",
                    "severity": 0.7
                },
            ]

            for issue in issues:
                session.run("""
                    MERGE (i:Issue {id: $id})
                    SET i.name = $name,
                        i.description = $description,
                        i.severity = $severity
                """, **issue)

            # Create COMPATIBLE_WITH relationships
            compatible_pairs = [
                ("group_dro", "class_balancing", 0.7, "Both address class imbalance from different angles"),
                ("group_dro", "undersampling", 0.5, "Can be combined but may reduce data"),
                ("class_balancing", "l2_regularization", 0.6, "Regularization helps with reweighting"),
                ("undersampling", "class_balancing", 0.4, "Redundant but not harmful"),
                ("importance_sampling", "l2_regularization", 0.6, "Complementary approaches"),
            ]

            for source, target, synergy, reason in compatible_pairs:
                session.run("""
                    MATCH (t1:Technique {id: $source}), (t2:Technique {id: $target})
                    MERGE (t1)-[r:COMPATIBLE_WITH]->(t2)
                    SET r.synergy = $synergy, r.reason = $reason
                """, source=source, target=target, synergy=synergy, reason=reason)

            # Create CONFLICTS_WITH relationships
            conflict_pairs = [
                ("group_dro", "strong_l2", 0.6, "Strong regularization suppresses group-specific learning"),
                ("undersampling", "importance_sampling", 0.4, "Conflicting data weighting strategies"),
                ("focal_loss", "class_balancing", 0.3, "Both modify loss weights - may conflict"),
            ]

            for source, target, severity, reason in conflict_pairs:
                session.run("""
                    MATCH (t1:Technique {id: $source}), (t2:Technique {id: $target})
                    MERGE (t1)-[r:CONFLICTS_WITH]->(t2)
                    SET r.severity = $severity, r.reason = $reason
                """, source=source, target=target, severity=severity, reason=reason)

            # Create IMPROVES relationships (technique -> metric)
            improves_pairs = [
                ("group_dro", "wga", 0.8),
                ("group_dro", "fairness_gap", 0.7),
                ("class_balancing", "wga", 0.6),
                ("class_balancing", "fairness_gap", 0.5),
                ("undersampling", "wga", 0.5),
                ("l2_regularization", "ood_accuracy", 0.5),
                ("l2_regularization", "robustness_gap", 0.4),
                ("domain_invariant", "ood_accuracy", 0.7),
                ("domain_invariant", "robustness_gap", 0.6),
                ("irm", "ood_accuracy", 0.7),
                ("focal_loss", "wga", 0.5),
            ]

            for tech, metric, effectiveness in improves_pairs:
                session.run("""
                    MATCH (t:Technique {id: $tech}), (m:Metric {id: $metric})
                    MERGE (t)-[r:IMPROVES]->(m)
                    SET r.effectiveness = $effectiveness
                """, tech=tech, metric=metric, effectiveness=effectiveness)

            # Create HAS_RISK relationships
            risk_pairs = [
                ("strong_l2", "underfitting", 0.7),
                ("undersampling", "overfitting", 0.4),
                ("group_dro", "overfitting", 0.3),
            ]

            for tech, issue, probability in risk_pairs:
                session.run("""
                    MATCH (t:Technique {id: $tech}), (i:Issue {id: $issue})
                    MERGE (t)-[r:HAS_RISK]->(i)
                    SET r.probability = $probability
                """, tech=tech, issue=issue, probability=probability)

            # Create CORRELATES_WITH relationships between metrics
            correlations = [
                ("wga", "fairness_gap", -0.8),  # Higher WGA = lower fairness gap
                ("ood_accuracy", "robustness_gap", -0.7),  # Higher OOD = lower gap
                ("id_accuracy", "ood_accuracy", 0.6),  # Generally correlated
            ]

            for m1, m2, correlation in correlations:
                session.run("""
                    MATCH (m1:Metric {id: $m1}), (m2:Metric {id: $m2})
                    MERGE (m1)-[r:CORRELATES_WITH]->(m2)
                    SET r.correlation = $correlation
                """, m1=m1, m2=m2, correlation=correlation)

        print("Base knowledge loaded successfully!")

    def query_compatible_techniques(self, technique: str) -> List[Dict]:
        """
        Find techniques compatible with the given technique.

        Args:
            technique: Technique ID (e.g., 'group_dro')

        Returns:
            List of compatible techniques with synergy scores
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t1:Technique {id: $tech})-[r:COMPATIBLE_WITH]->(t2:Technique)
                RETURN t2.id AS technique, t2.name AS name,
                       r.synergy AS synergy, r.reason AS reason
                ORDER BY r.synergy DESC
            """, tech=technique)
            return [dict(record) for record in result]

    def query_conflicting_techniques(self, technique: str) -> List[Dict]:
        """
        Find techniques that conflict with the given technique.

        Args:
            technique: Technique ID

        Returns:
            List of conflicting techniques with severity scores
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t1:Technique {id: $tech})-[r:CONFLICTS_WITH]->(t2:Technique)
                RETURN t2.id AS technique, t2.name AS name,
                       r.severity AS severity, r.reason AS reason
                ORDER BY r.severity DESC
            """, tech=technique)
            return [dict(record) for record in result]

    def query_techniques_for_metric(
        self,
        metric: str,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Find techniques that improve a specific metric.

        Args:
            metric: Metric ID (e.g., 'wga', 'ood_accuracy')
            threshold: Minimum effectiveness threshold

        Returns:
            List of techniques with effectiveness scores
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Technique)-[r:IMPROVES]->(m:Metric {id: $metric})
                WHERE r.effectiveness >= $threshold
                RETURN t.id AS technique, t.name AS name,
                       t.category AS category, t.description AS description,
                       r.effectiveness AS effectiveness
                ORDER BY r.effectiveness DESC
            """, metric=metric, threshold=threshold)
            return [dict(record) for record in result]

    def query_strategy_risks(self, strategy_config: Dict) -> List[Dict]:
        """
        Identify potential risks for a strategy configuration.

        Args:
            strategy_config: Strategy configuration dict

        Returns:
            List of identified risks
        """
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

                for record in result:
                    risks.append({
                        "type": "conflict",
                        "tech1": record["tech1"],
                        "tech2": record["tech2"],
                        "severity": record["severity"],
                        "reason": record["reason"]
                    })

            # Check for known issues with individual techniques
            result = session.run("""
                MATCH (t:Technique)-[r:HAS_RISK]->(i:Issue)
                WHERE t.id IN $techs
                RETURN t.name AS technique, t.id AS tech_id,
                       i.name AS issue, i.description AS description,
                       r.probability AS probability
            """, techs=techniques)

            for record in result:
                risks.append({
                    "type": "risk",
                    "technique": record["technique"],
                    "issue": record["issue"],
                    "description": record["description"],
                    "probability": record["probability"]
                })

        return risks

    def _extract_techniques(self, config: Dict) -> List[str]:
        """
        Extract technique IDs from a strategy configuration.

        Args:
            config: Strategy configuration dict

        Returns:
            List of technique IDs
        """
        techniques = []

        if config.get("use_group_dro"):
            techniques.append("group_dro")

        if config.get("class_weight") == "balanced":
            techniques.append("class_balancing")

        if config.get("undersample_majority"):
            techniques.append("undersampling")

        l2_c = config.get("l2_C", 1.0)
        if l2_c != 1.0:
            if l2_c <= 0.1:
                techniques.append("strong_l2")
            else:
                techniques.append("l2_regularization")

        return techniques

    def add_experiment_outcome(self, config: Dict, metrics: Dict):
        """
        Record an experiment outcome in the knowledge graph.

        Args:
            config: Strategy configuration
            metrics: Experiment metrics (accuracy, WGA, etc.)
        """
        exp_id = f"exp_{config.get('name', 'unknown')}_{hash(str(config)) % 10000}"

        with self.driver.session() as session:
            # Create experiment node
            session.run("""
                MERGE (e:Experiment {id: $exp_id})
                SET e.name = $name,
                    e.config = $config,
                    e.wga = $wga,
                    e.ood_acc = $ood_acc,
                    e.id_acc = $id_acc,
                    e.timestamp = datetime()
            """,
                exp_id=exp_id,
                name=config.get("name", "unknown"),
                config=str(config),
                wga=metrics.get("worst_group_accuracy", 0),
                ood_acc=metrics.get("ood_accuracy", metrics.get("accuracy", 0)),
                id_acc=metrics.get("id_accuracy", 0)
            )

            # Link to techniques used
            techniques = self._extract_techniques(config)
            for tech in techniques:
                session.run("""
                    MATCH (e:Experiment {id: $exp_id}), (t:Technique {id: $tech})
                    MERGE (e)-[:USED]->(t)
                """, exp_id=exp_id, tech=tech)

    def get_all_techniques(self) -> List[Dict]:
        """Get all techniques in the knowledge graph."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Technique)
                RETURN t.id AS id, t.name AS name, t.category AS category,
                       t.description AS description, t.effectiveness AS effectiveness
                ORDER BY t.effectiveness DESC
            """)
            return [dict(record) for record in result]

    def get_technique_details(self, technique_id: str) -> Optional[Dict]:
        """Get detailed information about a technique."""
        with self.driver.session() as session:
            # Get technique info
            result = session.run("""
                MATCH (t:Technique {id: $id})
                RETURN t.id AS id, t.name AS name, t.category AS category,
                       t.description AS description, t.paper AS paper,
                       t.effectiveness AS effectiveness
            """, id=technique_id)

            record = result.single()
            if not record:
                return None

            details = dict(record)

            # Get compatible techniques
            compat_result = session.run("""
                MATCH (t:Technique {id: $id})-[r:COMPATIBLE_WITH]->(t2:Technique)
                RETURN t2.name AS name, r.synergy AS synergy
            """, id=technique_id)
            details["compatible_with"] = [dict(r) for r in compat_result]

            # Get conflicting techniques
            conflict_result = session.run("""
                MATCH (t:Technique {id: $id})-[r:CONFLICTS_WITH]->(t2:Technique)
                RETURN t2.name AS name, r.severity AS severity
            """, id=technique_id)
            details["conflicts_with"] = [dict(r) for r in conflict_result]

            # Get metrics it improves
            improves_result = session.run("""
                MATCH (t:Technique {id: $id})-[r:IMPROVES]->(m:Metric)
                RETURN m.name AS metric, r.effectiveness AS effectiveness
            """, id=technique_id)
            details["improves"] = [dict(r) for r in improves_result]

            return details

    def format_kg_context(self, strategy_config: Dict) -> str:
        """
        Format knowledge graph insights for LLM prompts.

        Args:
            strategy_config: Current strategy configuration

        Returns:
            Formatted context string
        """
        context = "## Knowledge Graph Insights\n\n"

        # Get techniques from config
        techniques = self._extract_techniques(strategy_config)

        if not techniques:
            return context + "No specific techniques detected in configuration.\n"

        # Get risks
        risks = self.query_strategy_risks(strategy_config)
        if risks:
            context += "### Potential Risks\n"
            for risk in risks:
                if risk["type"] == "conflict":
                    context += f"- **Conflict**: {risk['tech1']} and {risk['tech2']} "
                    context += f"(severity: {risk['severity']:.1f})\n"
                    context += f"  Reason: {risk['reason']}\n"
                else:
                    context += f"- **{risk['issue']}** for {risk['technique']} "
                    context += f"(probability: {risk['probability']:.1f})\n"
            context += "\n"

        # Get compatible techniques for improvement
        context += "### Recommended Additions\n"
        for tech in techniques:
            compatible = self.query_compatible_techniques(tech)
            for compat in compatible[:2]:
                if compat["technique"] not in techniques:
                    context += f"- Consider adding **{compat['name']}** "
                    context += f"(synergy with {tech}: {compat['synergy']:.1f})\n"

        return context

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")


def test_knowledge_graph():
    """CLI function to test the knowledge graph."""
    print("=" * 50)
    print("Prometheus Lab - Knowledge Graph Test")
    print("=" * 50)

    try:
        kg = FairnessKnowledgeGraph()

        # Load base knowledge
        kg.load_base_knowledge()

        # Test queries
        print("\n--- Techniques for improving WGA ---")
        techniques = kg.query_techniques_for_metric("wga")
        for t in techniques:
            print(f"  {t['name']}: effectiveness={t['effectiveness']:.2f}")

        print("\n--- Techniques compatible with Group DRO ---")
        compatible = kg.query_compatible_techniques("group_dro")
        for c in compatible:
            print(f"  {c['name']}: synergy={c['synergy']:.2f}")

        print("\n--- Risk analysis for DRO + strong L2 ---")
        risks = kg.query_strategy_risks({
            "use_group_dro": True,
            "l2_C": 0.1,
            "class_weight": "balanced"
        })
        for r in risks:
            print(f"  {r['type']}: {r}")

        kg.close()

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Neo4j is running:")
        print("  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j")


if __name__ == "__main__":
    test_knowledge_graph()
