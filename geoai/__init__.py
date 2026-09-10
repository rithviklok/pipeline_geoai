"""
pipeline_geoai.geoai — Core GeoAI geocoding module.

Refactored from the original GeoAI Pipeline scripts:
  - train_geoai_geocoder.py (5,187 lines) → trainer.py
  - Inference.py (9,455 lines) → inferencer.py

Sub-modules:
  - columns     : GIS column auto-mapping & standardisation
  - embeddings  : SentenceTransformer + FAISS utilities
  - knowledge_base : KB persistence, caching, validation
  - trainer     : GeoAI training engine
  - inferencer  : GeoAI inference engine
"""
