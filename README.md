# Urban Sound 8K Audio Classification

This project aims to classify audio files from the Urban Sound 8K dataset using machine learning techniques. The code extracts audio features, specifically Mel-Frequency Cepstral Coefficients (MFCCs), to train a neural network for audio classification.

## Dataset

The metadata for the Urban Sound 8K dataset is loaded from the `metadata/UrbanSound8K.csv` file. The dataset contains various sound classes, and its imbalance is checked using `metadata['class'].value_counts()`.

## Feature Extraction

- Librosa library is used to load and process audio files.
- Audio features, specifically 40 MFCCs, are extracted and plotted for visualization.

## Data Processing

- Audio files are processed to extract features, and the results are stored in a DataFrame.
- Features are extracted for each audio file using the `feature_extracted` function.

## Model Training

- A neural network model is created using TensorFlow and Keras.
- The model consists of multiple layers with activation functions and dropout for regularization.
- Categorical cross-entropy is used as the loss function, and Adam optimizer is employed for training.

## Training the Model

- The model is trained on the extracted features with a split of 80% training and 20% testing data.
- Training progress is saved using ModelCheckpoint.

```python
# Example:
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer])

## **Prediction**
A sample audio file (audio/fold1/101415-3-0-3.wav) is used for prediction.
The trained model predicts the class label of the audio file.
