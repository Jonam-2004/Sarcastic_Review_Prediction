# Sarcasm Detection Using BiLSTM and BERT

This project focuses on detecting sarcasm in text data using deep learning techniques, including Bidirectional LSTM (BiLSTM) and BERT models. The goal is to classify text as sarcastic or non-sarcastic based on sentiment and contextual shifts.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [BiLSTM Model](#bilstm-model)
  - [BERT Model](#bert-model)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

---

## Overview
This project implements two neural network architectures:
1. **BiLSTM**: A Bidirectional LSTM model to capture sequential patterns in text.
2. **BERT**: A transformer-based model to leverage pre-trained embeddings for sarcasm detection.

The models are trained and evaluated on a text dataset to determine their performance in sarcasm detection tasks.

---

## Dataset
The dataset used in this project is loaded from a CSV file (`Final2.csv`) with the following columns:
- `label`: Indicates sarcasm (1) or non-sarcasm (0).
- `text`: Contains the text data for classification.

### Data Preparation
- Dropped unnecessary columns.
- Upsampled the minority class to handle data imbalance.
- Split the dataset into training and testing sets (70:30).

---

## Methodology

### Data Preprocessing
- Text tokenization using **BERT Tokenizer**.
- Padding and truncation to ensure uniform sequence length.
- Sentiment analysis using **TextBlob** for preprocessing and feature extraction.

### BiLSTM Model
- Input text is tokenized and embedded using an `Embedding` layer.
- A **Bidirectional LSTM** captures sequential patterns.
- A dense layer with a sigmoid activation is used for binary classification.
- Class weights are applied to address data imbalance.

### BERT Model
- Used `TFBertModel` from Hugging Face.
- Leveraged pre-trained BERT embeddings for text representation.
- Added a dropout layer for regularization and a dense layer for classification.

---

## Evaluation
The models are evaluated using the following metrics:
- **Accuracy**
- **F1-Score**
- **Confusion Matrix**
- **Classification Report**

### Visualizations
- Distribution of labels in the dataset.
- Heatmap of the confusion matrix.

---

## Usage

### Requirements
Install the required libraries:
```bash
pip install pandas numpy scikit-learn transformers tensorflow textblob seaborn matplotlib torch
```

### Run the Project
1. Load the dataset:
   ```python
   csv_path = "D:/Dataset/Final2.csv"
   ```
2. Train the BiLSTM model:
   ```python
   bi_lstm_model.fit(X_train_encoded['input_ids'], y_train, epochs=1, batch_size=64, class_weight=class_weight_dict)
   ```
3. Train the BERT model:
   ```python
   bert_classifier.fit([X_train_encoded['input_ids'], X_train_encoded['attention_mask']], y_train_repeated, epochs=1, batch_size=16, class_weight=class_weight_dict)
   ```
4. Evaluate and classify text inputs:
   ```python
   user_input = "Enter your text here"
   ```

---

## Results
- **BiLSTM Model**
  - Accuracy: 99.12%
  - F1-Score: 99.12%
- **BERT Model**
  - Accuracy: TBD
  - F1-Score: TBD

---

## Future Enhancements
- Fine-tune BERT on the specific sarcasm detection dataset.
- Experiment with transformer-based models like **RoBERTa** or **DistilBERT**.
- Incorporate multi-modal sarcasm detection (e.g., text and images).

---

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for review.

---

## Contact
For questions or suggestions, feel free to reach out!

**Author**: [Manoj S](https://github.com/Jonam-2004)

