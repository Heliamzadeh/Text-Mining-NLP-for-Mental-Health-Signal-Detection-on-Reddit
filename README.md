# Text-Mining-NLP-for-Mental-Health-Signal-Detection-on-Reddit
Text preprocessing pipeline (tokenization, lemmatization, noise removal)  TF-IDF + Word2Vec embeddings  Classical ML classifiers (LR, SVM, RF, AdaBoost)  Stratified cross-validation + hyperparameter tuning  Sentiment analysis (VADER) as baseline  Topic modeling (LDA + coherence analysis)  Temporal trend analysis

## Overview
This project applies advanced Natural Language Processing (NLP) and text mining techniques to identify mental health signals—specifically stress and depression—from social media discussions. Using Reddit data, the analysis integrates classical NLP representations, supervised machine learning models, sentiment analysis, and topic modeling to detect patterns in student well-being discourse.

The study focuses on posts from the McGill University subreddit and demonstrates how social media data can support early identification of high-stress academic periods and emerging mental health concerns.

## Problem Statement
University students increasingly express stress and mental health challenges through online forums, yet these signals often remain unstructured and difficult to quantify. This project addresses the problem of:
- Detecting stress and depression signals in free-text social media data
- Comparing classical NLP feature representations and classifiers
- Identifying temporal and thematic patterns related to academic stress cycles

## Data Sources
- **Training datasets**
  - Depression detection dataset (Kaggle, 7,732 posts)
  - Stress detection dataset from social media articles (5,557 posts)
- **Inference dataset**
  - McGill subreddit posts scraped using BeautifulSoup (≈2,800 posts)

After preprocessing and deduplication, 2,592 clean subreddit posts were used for downstream analysis.

## NLP Pipeline
The text processing and modeling pipeline includes:

### Text Preprocessing
- Tokenization
- Stop-word removal
- Lemmatization
- Removal of URLs, emojis, symbols, and non-alphabetic tokens

### Feature Representation
- **TF-IDF**
  - Vocabulary capped at top 3,000 terms
  - Terms appearing in <2 documents or >95% of documents removed
- **Word2Vec**
  - Dense semantic embeddings for contextual similarity analysis

### Modeling
Supervised classifiers were trained and evaluated using stratified cross-validation:
- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest
- AdaBoost

Hyperparameter tuning and 5-fold StratifiedKFold cross-validation ensured robust performance estimation.

### Baseline Sentiment Analysis
- VADER sentiment analysis used as a lexical baseline
- Compared against supervised classifiers to assess signal alignment and noise sensitivity

## Results
### Classification Performance
- **SVM + TF-IDF** achieved **96.27% test accuracy** for depression detection
- **Random Forest + Word2Vec** achieved **82.99% test accuracy** for stress detection

### Subreddit Analysis Insights
- Word2Vec embeddings showed stronger correlation with negative mental health posts
- After filtering positive posts:
  - Depression incidence ≈ 12%
  - Stress incidence ≈ 14%
- Peak mental health signal periods aligned with academic stress cycles:
  - April–May (final exams)
  - September–October (midterms)

## Topic Modeling
Latent Dirichlet Allocation (LDA) was applied to identify dominant themes:

- **Depression-related topics**
  - Academic overload, emotional distress, motivation loss, social support
  - Coherence score: 0.35 (broader thematic overlap)
- **Stress-related topics**
  - Coursework, exams, grades, performance pressure
  - Coherence score: 0.51 (more focused thematic structure)

## Key Takeaways
- Classical NLP techniques combined with robust ML models remain highly effective for mental health signal detection
- Word embedding methods capture emotional nuance beyond lexical sentiment
- Temporal and topic-level analysis provides actionable insights for academic support planning

## Tools & Technologies
- **Languages:** Python
- **NLP:** TF-IDF, Word2Vec, VADER, LDA
- **ML:** scikit-learn (Logistic Regression, SVM, Random Forest, AdaBoost)
- **Evaluation:** Stratified cross-validation, accuracy metrics
- **Data Collection:** BeautifulSoup
- **Visualization:** Matplotlib, Seaborn

## Future Work
- Extend analysis to additional universities
- Incorporate transformer-based models (BERT) for contextual classification
- Improve topic interpretability with dynamic topic modeling
- Explore causal relationships between academic events and mental health signals

## References
See full project report for methodology details.

