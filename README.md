# Cyberbullying Detection using Hybrid Text and Numeric Features

This notebook demonstrates a deep learning approach to detect cyberbullying using a combination of text data and numeric features. The project utilizes a dataset containing communication data between users, including message content, aggression counts, intent to harm scores, and demographic information.

## Dataset
The dataset is loaded from the following CSV files, which are assumed to be located at `/content/Predicting-Cyberbullying-on-Social-Media--A-Machine-Learning-Approach/dataset/`:
- `6. CB_Labels.csv`: Contains cyberbullying labels and related metrics.
- `1. users_data.csv`: Contains user demographic information.
- `5. Communication_Data_Among_Users.csv`: Contains communication details.

## Notebook Structure
The notebook is organized into the following sections:

1.  **Library Imports**: Imports necessary libraries for data manipulation, visualization, and deep learning.
2.  **Download NLTK assets**: Downloads required NLTK data for text preprocessing.
3.  **Define Constant Variables**: Sets up constant variables for model parameters.
4.  **Load Datasets**: Loads the three CSV files into pandas DataFrames.
5.  **Merge Demographic and CB_Labels**: Merges user demographic information from `users_df` into `cb_labels_df`.
6.  **Remove emojis, special characters, URLs, newline, etc**: Defines and applies functions to clean the message text in `communication_df`.
7.  **Remove Stopwords**: Defines and applies functions to remove stopwords from the cleaned text.
8.  **Lemmatization**: Applies lemmatization to the text data.
9.  **Merge required columns**: Merges `communication_df` with selected columns from `cb_labels_df` and creates `clean_com_df`.
10. **Compute 95th percentile length**: Calculates the 95th percentile of sentence lengths in `clean_com_df['Message3']` to determine `MAX_LEN`.
11. **Compute aggression_ratio**: Calculates the ratio of aggressive messages to total messages and adds it as a new column to `clean_com_df`.
12. **Normalize Numeric Features**: Standardizes the numeric features (`Total_messages`, `Aggressive_Count`, `Intent_to_Harm`) from `clean_com_df2` using `StandardScaler`.
13. **Split Dataset**: Splits the processed text data (`clean_com_df2['Message3']`), normalized numeric features, and labels (`clean_com_df2['Label']`) into training and validation sets. Creates TensorFlow Datasets (`train_ds`, `val_ds`).
14. **Text Vectorization**: Creates and adapts a `TextVectorization` layer (`vectorize_layer`) to the text data in `train_ds` to create a vocabulary (`VOCAB_SIZE`).
15. **Define Model**: Defines two model architectures:
    -   **LSTM**: A basic LSTM model (`lstm_model`) combining vectorized text and normalized numeric features.
    -   **Bidirectional LSTM**: A Bidirectional LSTM model with a custom Attention mechanism (`bilstm_attention_model`), combining vectorized text and normalized numeric features.
16. **Train Model**: Trains the defined models using `train_ds` and `val_ds`, with a class weight dictionary (`class_weight_dict`) and callbacks (`callbacks_dict`).
17. **Evaluate Model**: Evaluates the trained models using `val_ds`, generates classification reports, and visualizes confusion matrices.

## Preprocessing Steps

The text data undergoes the following preprocessing steps:
-   Removal of emojis and special unicode characters using `strip_emoji`.
-   Removal of dates and multiple spaces using `remove_dates_and_mult_spaces`.
-   Removal of newlines, links, non-ASCII characters, and most punctuation using `strip_all_entities`.
-   Cleaning of hashtags using `clean_hashtags`.
-   Filtering of words containing special characters using `filter_chars`.
-   Removal of custom and NLTK standard stopwords using `remove_stopwords_one` and `remove_stopwords`.
-   Lemmatization using `lemmatize_text`.
-   Padding punctuation with spaces using `pad_punctuation`.

The numeric features (`Total_messages`, `Aggressive_Count`, `Intent_to_Harm`) are standardized using `StandardScaler`.

## Models

Two deep learning models are implemented and evaluated:

-   **LSTM Model**:
    -   Takes text input (vectorized) and numeric input.
    -   Uses an Embedding layer for text.
    -   Processes text with an LSTM layer.
    -   Concatenates the LSTM output with numeric features.
    -   Uses Dense layers for classification.
    -   Compiled with Adam optimizer, BinaryFocalCrossentropy loss, and F1Score, Precision, and Recall metrics with a threshold of 0.3.

-   **Bidirectional LSTM with Attention Model**:
    -   Similar input layers and Embedding as the LSTM model.
    -   Uses a Bidirectional LSTM layer for text processing, capturing dependencies in both directions.
    -   Applies a custom `AttentionLayer` to the BiLSTM output to focus on important parts of the sequence.
    -   Concatenates the context vector from the Attention layer with numeric features.
    -   Uses Dense layers with L2 regularization and Dropout for classification.
    -   Compiled with Adam optimizer, BinaryFocalCrossentropy loss, and F1Score, Precision, and Recall metrics with a threshold of 0.3.

## Callbacks
The following Keras callbacks are used during model training, monitoring `val_f1_score`:
-   `ModelCheckpoint`: Saves the best model weights to `/content/drive/MyDrive/Project/checkpoint/checkpoint.weights.h5`.
-   `EarlyStopping`: Stops training if `val_f1_score` does not improve for 5 epochs.
-   `ReduceLROnPlateau`: Reduces the learning rate if `val_f1_score` does not improve for 5 epochs.
-   `TensorBoard`: Logs training progress to `/content/drive/MyDrive/Project/logs`.

## Evaluation
The models are evaluated using:
-   Classification Report: Provides precision, recall, and F1-score for each class.
-   Confusion Matrix: Visualizes the true vs. predicted labels using `seaborn.heatmap`.

The F1Score metric is used with a threshold of 0.3 for evaluation.

## How to Run
1. Ensure the dataset files are located at `/content/Predicting-Cyberbullying-on-Social-Media--A-Machine-Learning-Approach/dataset/`.
2. Ensure you have a directory at `/content/drive/MyDrive/Project/` for saving model checkpoints and logs.
3. Run the notebook cells sequentially.

## Dependencies
The notebook requires the following libraries:
-   pandas
-   numpy
-   matplotlib
-   seaborn
-   tensorflow
-   scikit-learn
-   imblearn
-   nltk

These dependencies are imported at the beginning of the notebook. Ensure you have these libraries installed in your environment. NLTK assets (`wordnet`, `punkt`, `stopwords`) are downloaded within the notebook.
