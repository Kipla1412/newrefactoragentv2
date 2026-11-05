"""Evaluation adapters — every outbound integration lives here.

Each adapter translates between one vendor or framework concern and the core
abstractions, so nothing in :mod:`evaluation.core` knows they exist:

==============================  ====================================================
Module                          Bridges core to
==============================  ====================================================
``agent_case_runner``           the real agents (AgentFactory + Session)
``deepeval_judge``              DeepEval's judge, backed by the framework LLMClient
``mlflow_scorer``               MLflow's built-in DeepEval scorers
``mlflow_tracker``              MLflow experiment tracking
``json_repository``             JSON persistence for reports and datasets
==============================  ====================================================
"""
