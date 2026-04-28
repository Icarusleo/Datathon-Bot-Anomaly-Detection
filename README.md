# Datathon Bot & Anomaly Detection

This repository contains the data analysis and anomaly detection pipeline for social media bot/manipulation detection. We explored several different methods and models to identify manipulative accounts and bot networks in the provided dataset.

## Repository Structure

- `eda_and_baseline_pipeline.ipynb`: Exploratory Data Analysis (EDA) and baseline anomaly detection pipeline using Isolation Forest. It extracts temporal, structural, NLP (lexical diversity), and coordination features to detect bot behavior.
- `unsupervised_pipeline.ipynb`: Advanced unsupervised pipeline featuring deep text embeddings (SentenceTransformers/CrossEncoder) for semantic rigidity analysis, combined with Isolation Forest.
- `unsupervised_pipeline2.ipynb`: An alternative iteration of the unsupervised pipeline, potentially with hyperparameter tuning or different feature engineering steps for improved bot detection.

## Methods Used

We applied a variety of techniques to capture different dimensions of bot behavior:
1. **Temporal Mechanics**: Measuring the variance and consistency of inter-arrival times between posts (bots tend to have low variance, posting at exact intervals).
2. **Coordination & Campaigns**: Identifying accounts that post simultaneously on the same platforms about the same themes.
3. **NLP & Lexical Diversity**: Using unique keyword ratios to identify copy-paste spam and keyword stuffing.
4. **Semantic Rigidity**: Using `sentence-transformers` (CrossEncoder/MiniLM) to embed text and calculate semantic similarity. Bots posting highly similar texts repeatedly are flagged.
5. **Anomaly Detection Models**:
   - **Isolation Forest**: Primary model used to isolate anomalous (manipulative) accounts based on the extracted features.
   - **Hybrid Isolation Forest**: Combines tabular features with PCA-reduced text embeddings for a more robust organic score.

## Note on Data & Models
The datasets (`.parquet`, `.html`) and trained model weights (`.pkl` files in the `models/` directory) are ignored via `.gitignore` due to their large size. To run the notebooks, ensure the appropriate datasets are downloaded and paths are updated in the notebooks.
