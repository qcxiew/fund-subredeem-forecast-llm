# fund-subredeem-forecast-llm
This repository contains code implementation in PyTorch with LLM-based financial feature engineering and an automated time-series forecasting auto/AI agent pipeline, for the task: Long-term Subscription and Redemption Prediction of Fund Products. This project builds an end-to-end pipeline that uses LLM-based financial features and a deep time-series model to forecast long-term fund subscription and redemption volumes.

- This project explicitly follows the key requirements of the AFAC2025 Problem 1 task (https://tianchi.aliyun.com/competition/entrance/532352/information):
   1. LLM-based feature engineering for financial signals
      - Financial features (e.g., product returns, market conditions, sector dynamics)  
        → must be constructed using large language models (e.g., Qwen3, Llama3, or similar open-source models).
      - Non-financial features (e.g., trading-day flags, calendar attributes)  
        → may be manually engineered using traditional methods.
   
   2. Comprehensive evaluation
      - Performance is evaluated using WMAPE (Weighted Mean Absolute Percentage Error).  
        Lower WMAPE indicates better forecasting performance.
      - Final assessment combines:
        - Prediction accuracy of subscription/redemption volumes; and
        - Quality and depth of LLM usage for feature construction.

- This repository implements end-to-end pipeline designed to satisfy these task requirements and to achieve competitive forecasting accuracy:
   1. auto/AI Agent and Automated Pipeline
      - This project is built around a lightweight auto/AI agent implemented in `fund_agent_service.py`.  The agent is responsible for automatically executing the full long-term subscription and redemption prediction workflow.
   
   2. How to run the agent (basic deployment)
      - The simplest way to run the project is to start the agent directly from the command line:
        bash directly run:
        ```bash
         python fund_agent_service.py
   
   3. Publish results
      - after model training, auto publish predict result and copy the result files from `tmp/result_pred/` back into the project-level `result_pred/` directory for easy access.

- Project Overview
   Long-term subscription and redemption prediction of fund products for AFAC2025 Problem Task, implemented in PyTorch with LLM-based financial feature engineering and an automated time-series forecasting pipeline.
   
   Goal in this project is to build a practical solution via auto/AI agent that:
   - Uses large language models (LLMs) to construct financial features, as required by the task;
   - Combines these features with classical and deep time-series models;
   - Produces competitive predictions of fund subscription and redemption volumes over a future horizon.

  Key Components and Their Roles
   - LLM (LLM-Embedding). The LLM is used purely as a feature encoder, turning fund profile text and news articles into dense semantic embeddings that describe each fund’s characteristics and current context.
   - Time-series model (BiLSTM + N-BEATS). The time-series model takes historical numerical series plus LLM embeddings and time features as input and outputs multi-step forecasts of future subscription and redemption volumes.
   - s2_feature_extraction.py. This script reads fund static information and uses Qwen3 embeddings to generate a fixed semantic vector for each fund.
   - s3_vectorstore_build.py. This script converts fund news (title + content) into embeddings with Qwen3 and stores them, along with metadata, in a persistent Chroma vectorstore.
   - s4_train_model_mm.py. This script builds the multi-modal time-series dataset, trains the BiLSTM + N-BEATS forecasting model with WMAPE loss, and saves the best model checkpoint for inference.
   - s6_eval_predict.py. This script loads prediction result files, computes WMAPE-based evaluation metrics, and ranks different runs to compare forecasting performance.
     
   From the service provider perspective, accurate subscription/redemption forecasts can support liquidity management decisions, reducing capital costs caused by platform pre-funding; and reveal capital flows across sectors, enabling targeted early-warning and operational strategies to mitigate AUM loss. From the user/investor perspective, these forecasts help institutions and investors prepare trades and lock positions in advance; reduce return frictions and improve users’ perceived return experience when combined with investment research and intervention strategies. This repository focuses on the code and modeling solution.




