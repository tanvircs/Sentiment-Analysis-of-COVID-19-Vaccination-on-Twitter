# Sentiment-Analysis-of-COVID-19-Vaccination-on-Twitter

## Overview
This project analyzes public sentiment on Twitter regarding COVID-19 vaccination using Natural Language Processing (NLP) and machine learning techniques. The data contains over 16,000 tweets classified into positive, neutral, and negative sentiments.

The analysis uses feature extraction methods such as Count Vectorizer, TF-IDF, Word2Vec, and Doc2Vec, and applies a K-Nearest Neighbors (KNN) classifier. The visual outputs, including loss, accuracy graphs, and word clouds, are included below.

## Datasets
The following datasets are used in this project:
- **final_data.csv**: The complete dataset of tweets with sentiment labels.
- **final_train.csv**: Training dataset for model training.
- **final_test.csv**: Test dataset for evaluating model performance.

Labels for sentiment:
- 0: Negative Sentiment
- 1: Positive Sentiment
- 2: Neutral Sentiment

## Methodology
- **Data Preprocessing**: Clean and prepare tweet data for feature extraction.
- **Feature Extraction**: Use methods like Count Vectorizer, TF-IDF, Word2Vec, and Doc2Vec to transform text into numerical features.
- **Modeling**: Train a KNN classifier to predict tweet sentiment.
- **Evaluation**: Assess model performance with accuracy and loss metrics on training and validation sets.

## Setup and Installation
### Prerequisites
Ensure you have Python installed. You can download it here.

### Installation
Clone the repository:
```
git clone <repository_url>
cd <repository_folder>
```
