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
1. #### Clone the repository:
```
git clone <repository_url>
cd <repository_folder>
```
2. **Install dependencies**: Install the required libraries using pip:
```
pip install -r requirements.txt
```

## Results and Visualizations

### Sentiment Distribution

![Sentiment Distribution](./output/Sentiment_Distribution.png)

---

### Training vs Test Data Distribution

![Train Test Distribution](./output/Train_Test_Distribution.png)

---

### Loss and Accuracy for Different Feature Extraction Methods

#### 1. **Count Vectorizer**:

- **Loss**:
  ![Loss Counter Vectorizer](./output/Loss_CounterVectorizer.png)

- **Accuracy**:
  ![Accuracy Counter Vectorizer](./output/Classifier_Accuracy_CounterVectorizer.png)

#### 2. **TF-IDF**:

- **Loss**:
  ![Loss TF-IDF](./output/Loss_TF_IDF.png)

- **Accuracy**:
  ![Accuracy TF-IDF](./output/Classifier_Accuracy_TF_IDF.png)

#### 3. **Doc2Vec**:

- **Loss**:
  ![Loss Doc2Feature](./output/Loss_Doc2Feature.png)

- **Accuracy**:
  ![Accuracy Doc2Feature](./output/Classifier_Accuracy_Doc2Feature.png)

#### 4. **Word2Vec**:

- **Loss**:
  ![Loss Word2Vec](./output/Loss_Word2Vec.png)

- **Accuracy**:
  ![Accuracy Word2Vec](./output/Classifier_Accuracy_Word2Vec.png)

---

### Word Clouds

#### Positive Sentiment Word Cloud

![Positive Word Cloud](./output/Positive_Word_Clouds.png)

#### Neutral Sentiment Word Cloud

![Neutral Word Cloud](./output/Neutral_Word_Clouds.png)

#### Negative Sentiment Word Cloud

![Negative Word Cloud](./output/Negative_Word_Clouds.png)

---

### Hashtags Analysis

#### Positive Hashtags

![Positive Hashtags](./output/Positive_Word_Hashtags.png)

#### Neutral Hashtags

![Neutral Hashtags](./output/Neutral_Word_Hashtags.png)

#### Negative Hashtags

![Negative Hashtags](./output/Negative_Word_Hashtags.png)

---

## Conclusion

This project successfully demonstrates the application of NLP and machine learning for sentiment analysis of COVID-19 vaccine-related tweets.

