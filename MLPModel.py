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


    # Define the MLP model with adjustable hidden layers
    def create_mlp_model(input_shape, num_classes, hidden_layers):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        for units in hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        model.compile(optimizer='sgd',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model


    # Model parameters
    input_shape = (X.shape[1],)
    num_classes = len(np.unique(y))

    # Different architectures to experiment with
    architectures = [
        #[64],  # 1 hidden layer
        #[128],  # 1 hidden layer
        #





#most trying a new parapmters ....

        #[64, 32],  # 2 hidden layers
       # [128, 64],  # 2 hidden layers
        [128, 64, 32],  # 3 hidden layers
       # [256, 128, 64] # 3 hidden layers
    ]

    comms_rounds = 10
    batch_size = 32

    # Results storage
    results = []

    # Iterate through each architecture
    for hidden_layers in architectures:
        print(f"\nTraining with architecture: {hidden_layers}")

        global_model = create_mlp_model(input_shape, num_classes, hidden_layers)
        client_training_times = [0] * 5  # 5 clients

        for round_num in range(comms_rounds):
            round_start_time = time.time()
            print(f'Communication Round: {round_num + 1}/{comms_rounds}')

            for client in range(5):
                client_start_time = time.time()
                client_indices = np.random.choice(X.shape[0], size=int(X.shape[0] * 0.2), replace=False)
                client_X = X[client_indices]
                client_y = y[client_indices]

                client_model = create_mlp_model(input_shape, num_classes, hidden_layers)
                client_model.set_weights(global_model.get_weights())

                # Measure client training time
                client_training_start_time = time.time()
                client_model.fit(client_X, client_y, epochs=10, batch_size=batch_size, verbose=0)
                client_training_time = time.time() - client_training_start_time

                # Update global model weights
                for layer, client_layer in zip(global_model.layers, client_model.layers):
                    new_weights = [(global_weight + client_weight) / 2 for global_weight, client_weight in
                                   zip(layer.get_weights(), client_layer.get_weights())]
                    layer.set_weights(new_weights)

                client_training_times[client] += client_training_time  # Accumulate client training time

            # Evaluate the global model
            eval_start_time = time.time()
            y_pred = np.argmax(global_model.predict(X), axis=1)
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            eval_time = time.time() - eval_start_time

            round_time = time.time() - round_start_time

        # Collect results
        total_training_time = sum(client_training_times)
        avg_precision = precision
        avg_recall = recall
        avg_f1 = f1

        results.append({
            'architecture': hidden_layers,
            'total_training_time': total_training_time,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1
        })

    # Print summary of results
    print("\nSummary of Results:")
    print(f"{'Architecture':<30} {'Total Training Time (s)':<25} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    for result in results:
        print(
            f"{str(result['architecture']):<30} {result['total_training_time']:<25.4f} {result['precision']:<10.4f} {result['recall']:<10.4f} {result['f1_score']:<10.4f}")