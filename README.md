# Toxic-comments-classifying-using-pre-trained-Toxic-BERT
Toxic Comment Classification
This project tackles multi-label toxic comment classification using the Jigsaw/Wikipedia dataset (~159K training comments). The goal is to classify each comment across six toxicity categories:
toxic
severe_toxic
obscene
threat
insult
identity_hate
Dataset Attribution
This work utilizes the dataset from the Kaggle competition:
Toxic Comment Classification Challenge
Hosted by Jigsaw / Conversation AI
The dataset was released as part of a Featured Prediction Competition and consists of Wikipedia comments labeled for multiple toxicity categories.
Approach
Three modeling strategies were implemented and compared:
1. Baseline — TF-IDF + Logistic Regression
TF-IDF with bigrams
5,000 features
One-vs-Rest classifier
Implemented using scikit-learn
2. Deep Learning — BiLSTM / BiLSTM + Attention
Implemented using TensorFlow / Keras
Pipeline:
Tokenization
Padding to 150 tokens
Embedding layer
Bidirectional LSTM
Optional custom Attention layer
Sigmoid output (multi-label classification)
3. Transformer — Toxic-BERT
Model: unitary/toxic-bert
Pre-trained specifically for toxic comment detection
Implemented using Hugging Face Transformers
Executed on Google Colab (Tesla T4 GPU)
Per-class threshold tuning applied
Text Preprocessing
Two cleaning pipelines were developed:
For BERT:
Lowercasing
URL masking
Email masking
IP masking
User mention masking
For baseline models:
All of the above
Punctuation removal
Repeated character normalization
Results
Model	Macro F1	Micro F1
BiLSTM	0.473	0.734
BiLSTM + Attention	0.426	0.723
Toxic-BERT (tuned)	0.863	0.902
Performance Highlights
BERT achieved +82.7% improvement in Macro F1 over BiLSTM.
AUC-ROC:
BiLSTM: 0.854
Toxic-BERT: 0.989
Tech Stack
Python
Pandas
scikit-learn
TensorFlow / Keras
Hugging Face Transformers
Google Colab (Tesla T4 GPU)
