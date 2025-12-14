# Audience Decode: Hybrid Behavioral Analysis

**Team:** [Arshat Kaipbergen / Alikhan Salimov / Anuar Abdumannapov]
**Course:** Machine Learning (2025/2026)

## 1. Introduction
This project analyzes the `viewer_interactions` dataset to uncover behavioral patterns within a large-scale streaming platform. Rather than simply predicting individual ratings, our goal was to "decode" the audience by identifying distinct user segments, discovering latent taste clusters, and determining the primary drivers of user enjoyment.

## 2. Methods
We employed a multi-stage machine learning pipeline combining unsupervised and supervised techniques:

1.  **Preprocessing:** Data was merged from `viewer_ratings`, `movies`, and `user_statistics`. We handled missing values by dropping incomplete rows to preserve SVD integrity and normalized rating scales.
2.  **Model 1 (Clustering):** We used **K-Means Clustering** ($k=4$) on user statistics (activity, average rating) to segment the audience into behavioral groups (e.g., "Power Users").
3.  **Model 2 (Pattern Discovery):** We used **Truncated SVD** (Matrix Factorization) to extract 10 latent factors from the sparse user-item matrix, identifying hidden genre preferences.
4.  **Model 3 (Hybrid Classification):** We trained a **Random Forest Classifier** to predict high ratings (4-5 stars).
    * **Features:** We engineered a hybrid feature set combining Metadata (`year`), User Behavior (`generosity`), and **Latent Taste** (the 10 SVD vectors).
    * **Tuning:** We utilized `RandomizedSearchCV` with 3-fold cross-validation to optimize hyperparameters. We specifically tuned `n_estimators` (finding 150 to be optimal) and `max_depth` (finding 20 to be optimal) to balance model complexity and prevent overfitting.

## 3. Experimental Design
To validate our findings rigorously:
* **Baseline Comparison:** We implemented a `DummyClassifier` (predicting the most frequent class) to establish a performance floor (Accuracy: 0.58). This validated that our model's accuracy was not due to class imbalance.
* **Metrics:**
    * *Elbow Method:* Used to determine the optimal $k$ for clustering.
    * *Accuracy & Feature Importance:* Used to evaluate the Random Forest.
    * *Qualitative Analysis:* Used to interpret the meaning of SVD factors.
* **Validation:** We used an 80/20 Train/Test split on a large sample (120k rows) to ensure robustness.

## 4. Results
### A. Latent Taste Factors (SVD)
The matrix factorization successfully uncovered hidden genres without explicit labels:
* **Factor 1:** Popular Blockbusters (e.g., *Pearl Harbor*, *Rain Man*)
* **Factor 2:** Suspense/Thrillers (e.g., *The General's Daughter*)
* **Factor 3:** Critical Dramas (e.g., *Terms of Endearment*)

### B. Hybrid Model Performance
Our tuned Hybrid Random Forest achieved an **accuracy of 74%**.
* **Baseline:** 58% (Random Guessing)
* **Lift:** +16% improvement over the baseline.

![Feature Importance](images/hybrid_importance.png)
*Figure 1: Feature Importance Analysis.*

The analysis reveals that **User Generosity** (0.34) is the primary driver of ratings, but the inclusion of **Latent Taste Factors** (like Factor 1) provided the critical nuance required to achieve high accuracy.

## 5.  Conclusions
We successfully decoded the audience by proving that ratings are a complex function of **Who you are** (Generosity) + **What you like** (SVD Factors).
* **Key Insight:** A user's historical rating behavior is a stronger predictor of enjoyment than the movie's popularity alone.
* **Future Work:** A cold-start strategy is needed for new users who lack the history required for SVD vector generation.
