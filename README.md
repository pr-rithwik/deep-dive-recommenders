# Deep Dive Recommenders: From Matrix Factorization to a Hybrid Two-Tower Model

## 🎯 Project Overview

This project is a comprehensive, comparative study of recommender system architectures, systematically building and evaluating three tiers of intelligence: non-personalized baselines, classic collaborative filtering, and a modern hybrid deep learning model. Using a rich dataset with movie content features (genres, tags), this project demonstrates the quantifiable performance lift achieved by incorporating content awareness into state-of-the-art neural network architectures.

The final deliverable is an interactive web application that allows users to compare the recommendations from different models side-by-side, providing an intuitive understanding of their underlying logic.

## ✨ Key Features

-   **Three Tiers of Intelligence:** Rigorous comparison of Content-Based, classic Matrix Factorization (SVD), and a hybrid Two-Tower deep learning model.
-   **Rich Feature Engineering:** Leverages movie genres and user-generated tags (`TfidfVectorizer`) to create a detailed "content fingerprint" for each movie.
-   **Modern Deep Learning Architecture:** Implements a state-of-the-art Two-Tower model in TensorFlow/Keras, learning from both user behavior and item content.
-   **Interactive Showcase:** A Streamlit/Flask web application for real-time recommendation comparison.
-   **Comprehensive Evaluation:** Measures performance using both regression metrics (RMSE, MAE) and ranking metrics (Precision@k, Recall@k).

## 🛠️ Tech Stack

-   **Data Processing:** Python, Pandas, NumPy, Scikit-learn
-   **Modeling:** TensorFlow (Keras), Scikit-learn
-   **Web Application:** Streamlit / Flask
-   **Analysis & Visualization:** Jupyter Notebooks, Matplotlib, Seaborn

## 📂 Project Structure

```
deep-dive-recommenders/
├── data/
│   ├── raw/              # Original dataset files (movies.csv, ratings.csv, etc.)
│   └── processed/        # Cleaned data and engineered features
├── notebooks/
│   ├── 01_EDA_and_Feature_Engineering.ipynb
│   ├── 02_Baseline_and_Content_Model.ipynb
│   ├── 03_Matrix_Factorization_Model.ipynb
│   └── 04_Two_Tower_Hybrid_Model.ipynb
├── src/
│   ├── data_processing.py
│   ├── models.py         # Model definitions (SVD, Two-Tower, etc.)
│   ├── train.py          # Scripts to train the models
│   └── evaluate.py       # Scripts to evaluate model performance
├── app/
│   ├── app.py            # The main Streamlit/Flask application file
│   └── assets/           # Saved models, feature matrices, etc. for the app
├── requirements.txt
└── README.md
```

## 📈 The Three Tiers of Intelligence: A Four-Model Comparison

This project evaluates four distinct models organized into three strategic tiers of intelligence. This structure tells a clear story of increasing sophistication, from simple baselines to a state-of-the-art hybrid model.

| Tier | Model | Logic | Data Used | Question Answered |
| :--- | :--- | :--- | :--- | :--- |
| **1: Baselines** | Content-Based Similarity | "You liked this movie, so you'll like movies with similar tags/genres." | Movie Content Only | *How well can we recommend without personalization?* |
| **2: Collaborative** | Matrix Factorization (SVD) | "Users similar to you liked these movies." | User-Movie Ratings Only | *How well can we recommend using only user behavior?* |
| **3: Hybrid** | Two-Tower Neural Network | "Let's learn your unique taste profile from your ratings and match it to movies based on their content." | User-Movie Ratings **+** Movie Content | *What is the performance lift from a content-aware hybrid model?* |

## 🚀 Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pr-rithwik/deep-dive-recommenders.git
    cd deep-dive-recommenders
    ```
2.  **Set up the environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Run the pipeline:**
    *   Execute the Jupyter notebooks `01` through `04` in order to process data, train, and evaluate all models.
4.  **Launch the interactive showcase:**
    ```bash
    cd app
    streamlit run app.py
    ```

## 📊 Expected Results

*(This section will be populated with the final comparison table)*

| Model | RMSE | Precision@10 | Recall@10 | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Popularity Baseline** | - | - | - | Non-personalized |
| **Content-Based** | - | - | - | Non-personalized |
| **Matrix Factorization (SVD)** | | | | Collaborative Only |
| **Two-Tower Hybrid Model** | | | | **Content + Collaborative** |

## 🔮 Future Work

-   **Incorporate Temporal Features:** Use the timestamp data to model evolving user tastes and the "freshness" of content.
-   **Advanced NLP on Tags:** Use pre-trained language models (e.g., Sentence-BERT) to create even richer embeddings from movie tags.
-   **Optimize for Ranking:** Experiment with ranking-specific loss functions (like BPR or Triplet Loss) to directly optimize the order of recommendations.
