"""
Entity Linking module for knowledge base linking.

This module provides entity linking capabilities for mapping biomedical entity mentions
to standardized ontology URIs using the entity-linkings library from NAIST.

Main classes:
    - ELEntityLinker: Main interface for entity linking inference and evaluation
    - EntityLinkingData: PyTorch Dataset wrapper for entity-linkings format data
    - DataConverter: Bidirectional conversion between GBIE and entity-linkings formats
    - EntityLinkingEvaluator: Evaluation metrics (strict and span-only)

Usage:
    from src.entityLinker import ELEntityLinker, EntityLinkingEvaluator
    
    linker = ELEntityLinker(config)
    linker.from_pretrained(model_name, entity_dict_path)
    predictions = linker.perform_inference(gbie_annotation)
    
    evaluator = EntityLinkingEvaluator()
    metrics = evaluator.evaluate(predictions["predicted_entities"], 
                                 gbie_annotation["entities"])
"""

from .ELEntityLinker import (
    ELEntityLinker,
    EntityLinkingData,
    EntityLinkingEvaluator,
    DataConverter,
)

__all__ = [
    "ELEntityLinker",
    "EntityLinkingData",
    "EntityLinkingEvaluator",
    "DataConverter",
]
