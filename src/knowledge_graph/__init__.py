"""
Knowledge Graph module for Prometheus Lab.

This module provides a Neo4j-based knowledge graph for:
- Technique relationships (compatibility, conflicts)
- Metric correlations
- Strategy risk analysis
- Experiment outcome learning

Usage:
    from src.knowledge_graph import FairnessKnowledgeGraph

    kg = FairnessKnowledgeGraph()
    kg.load_base_knowledge()

    # Query techniques for improving WGA
    techniques = kg.query_techniques_for_metric("wga")

    # Check compatibility
    compatible = kg.query_compatible_techniques("group_dro")

    # Get strategy risks
    risks = kg.query_strategy_risks(strategy_config)

    kg.close()
"""

from .graph import FairnessKnowledgeGraph

__all__ = ["FairnessKnowledgeGraph"]
