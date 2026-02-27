import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# Define file paths for your dataset
# Define file paths for your dataset
# Define file paths for your dataset
file_paths = [

    'client_data_iid_1.csv',
    'client_data_iid_2.csv',
    'client_data_iid_3.csv',
   'client_data_iid_4.csv',
    'client_data_iid_5.csv'
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


    # Create the RNN model
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

    comms_rounds = 10
    batch_size = 32
    results = []

    print("\nTraining RNN model...")
    global_model = create_rnn_model(input_shape, num_classes)

    for round_num in range(comms_rounds):
        round_start_time = time.time()
        print(f'Communication Round: {round_num + 1}/{comms_rounds}')

        client_training_times = [0] * 5  # CL: Initialize array to track client training times

        for client in range(5):  # CL: Loop over each client
            client_start_time = time.time()  # CL: Start timer for client training
            client_indices = np.random.choice(X.shape[0], size=int(X.shape[0] * 0.2), replace=False)
            client_X = X[client_indices]
            client_y = y[client_indices]

            # Create a client model; if client is 0, use baseline settings
            client_model = create_rnn_model(input_shape, num_classes,
                                            use_baseline=(client == 0))  # CL: Baseline for client 0
            client_model.set_weights(global_model.get_weights())  # CL: Initialize client model with global weights

            # Train the client model
            client_model.fit(client_X, client_y, epochs=10, batch_size=batch_size, verbose=0)  # CL: Train client model

            # Update global model weights
            for layer, client_layer in zip(global_model.layers, client_model.layers):
                new_weights = [(global_weight + client_weight) / 2 for global_weight, client_weight in
                               zip(layer.get_weights(), client_layer.get_weights())]  # CL: Average weights
                layer.set_weights(new_weights)  # CL: Update global model weights

            client_training_times[client] += time.time() - client_start_time  # CL: Record client training time

        # Evaluate the global model
        y_pred = np.argmax(global_model.predict(X), axis=1)  # CL: Get predictions from the global model
        precision = precision_score(y, y_pred, average='weighted')  # CL: Calculate precision
        recall = recall_score(y, y_pred, average='weighted')  # CL: Calculate recall
        f1 = f1_score(y, y_pred, average='weighted')  # CL: Calculate F1 score

        total_training_time = sum(client_training_times)  # CL: Calculate total training time for clients

        # Store results
        results.append({
            'total_training_time': total_training_time,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

        round_time = time.time() - round_start_time
        print(f"Round time: {round_time:.4f} seconds")

        # Print summary for the current round
        print(
            f"End of communication round {round_num + 1}. Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Print summary of results
    print("\nSummary of Results:")
    print(f"{'Total Training Time (s)':<25} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    for result in results:
        print(
            f"{result['total_training_time']:<25.4f} {result['precision']:<10.4f} {result['recall']:<10.4f} {result['f1_score']:<10.4f}")