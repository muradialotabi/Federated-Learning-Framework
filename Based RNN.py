import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Define file paths for your dataset
file_paths = [
    'client_data_25_50_100_200_client_1.csv',
    'client_data_25_50_100_200_client_2.csv'
]


def load_data(file_paths):
    start_time = time.time()
    df_list = []

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                df_list.append(df)
            else:
                print(f"Warning: '{file_path}' is empty.")
        except Exception as e:
            print(f"Error reading '{file_path}': {e}")

    combined_df = pd.concat(df_list, ignore_index=True) if df_list else None
    load_time = time.time() - start_time
    print(f"Data loading time: {load_time:.4f} seconds")
    return combined_df


data = load_data(file_paths)

if data is not None:
    label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 9: 5}
    data['label'] = data['label'].map(label_mapping)

    # Split features and labels
    X = data.drop('label', axis=1).values
    y = data['label'].values

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reshape data for RNN [samples, time_steps, features]
    time_steps = 1  # Adjust based on your specific learning problem
    X = X.reshape((X.shape[0], time_steps, X.shape[1]))


    # Create RNN model
    def create_rnn_model(input_shape, num_classes):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.SimpleRNN(128, activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=True))
        model.add(tf.keras.layers.SimpleRNN(32, activation='relu'))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        # Compile with SGD optimizer
        model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.01),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model


    input_shape = (time_steps, X.shape[2])  # (time_steps, features)
    num_classes = len(np.unique(y))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the global model
    print("\nTraining RNN model...")
    global_model = create_rnn_model(input_shape, num_classes)

    # Train the model on the centralized dataset
    start_time = time.time()
    global_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.4f} seconds")

    # Evaluate the model
    print("\nEvaluating the model...")
    results = global_model.evaluate(X_test, y_test, verbose=0)
    accuracy = results[1]  # The second element returned is accuracy

    # Make predictions
    y_pred = np.argmax(global_model.predict(X_test), axis=1)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print evaluation results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Save the model
    global_model.save('rnn_model.h5')
    print("Model saved as 'rnn_model.h5'")