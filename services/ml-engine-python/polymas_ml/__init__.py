"""
Polymas ML Engine - Multi-label ensemble for diabetes risk prediction.

Serves predictions via gRPC with base learners (XGBoost, CatBoost, LightGBM),
score normalization (Platt scaling), weighted voting, and explainability.
"""
