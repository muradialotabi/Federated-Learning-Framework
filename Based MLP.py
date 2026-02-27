import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# Define file paths for your dataset
file_paths = [
    'client_data_25_50_100_200_client_1.csv',
    'client_data_25_50_100_200_client_2.csv',
    'client_data_25_50_100_200_client_3.csv',
    'client_data_25_50_100_200_client_4.csv',
    'client_data_25_50_100_200_client_5.csv'
]

# Load and preprocess data
def load_data(file_paths):
    start_time = time.time()
    df_list = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                df_list.append(df)
        except Exception as e:
            print(f"Error reading '{file_path}': {e}")

    combined_df = pd.concat(df_list, ignore_index=True) if df_list else None
    load_time = time.time() - start_time
    print(f"Data loading time: {load_time:.4f} seconds")
    return combined_df

data = load_data(file_paths)

if data is not None:
    # Remap labels if necessary
    label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 9: 5}
    data['label'] = data['label'].map(label_mapping)

    # Split into features and labels
    X = data.drop('label', axis=1).values
    y = data['label'].values

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define the MLP model
    def create_mlp_model(input_shape, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='sgd',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    # Initialize the MLP model
    input_shape = (X.shape[1],)
    num_classes = len(np.unique(y))
    mlp_model = create_mlp_model(input_shape, num_classes)

    # Train the MLP model
    start_time = time.time()
    mlp_model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    training_time = time.time() - start_time

    # Evaluate the MLP model
    y_pred = np.argmax(mlp_model.predict(X), axis=1)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    print("MLP Model Performance:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MLP Training Time: {training_time:.4f} seconds")