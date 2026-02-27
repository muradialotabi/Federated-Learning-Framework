import os
import pandas as pd

# Define file paths
file_paths = [
    'ConstPosFullPathes.csv',
    'ConstPosOffsetFullPathes.csv',
    'EventalStopFullPathes.csv',
    'RandomPosFullPathes.csv',
    'RandomPosOffsetFullPathes.csv'
]

# Load and preprocess data
def load_data(file_paths):
    df_list = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Data from '{file_path}': {df.shape[0]} rows, {df.columns.tolist()}")
            if not df.empty:
                df_list.append(df)
                print(f"Successfully read '{file_path}' with {len(df)} rows.")
            else:
                print(f"Warning: '{file_path}' is empty.")
        else:
            print(f"Error: File '{file_path}' does not exist.")

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        print(f"Combined DataFrame shape: {combined_df.shape}")
        return combined_df
    else:
        print("No valid data loaded.")
        return None

def iid_partition_by_sender(df, num_clients):
    partitions = []
    classes = df['sender'].unique()

    for client in range(num_clients):
        client_data = pd.DataFrame()
        for class_label in classes:
            class_data = df[df['sender'] == class_label]
            n_samples = min(200, len(class_data))  # Adjust this number as needed
            client_data = pd.concat([client_data, class_data.sample(n=n_samples, random_state=client)])
        partitions.append(client_data.sample(frac=1).reset_index(drop=True))  # Shuffle the client data

    return partitions

# Main execution
data = load_data(file_paths)

if data is not None:
    num_clients = 5# Set the number of clients

    # Partitioning data IID by sender
    client_data = iid_partition_by_sender(data, num_clients)

    # Save each client's data to a CSV file
    for client in range(num_clients):
        client_file_path = f'client_data_iid_{client + 1}.csv'
        client_data[client].to_csv(client_file_path, index=False)
        print(f"Saved data for client {client + 1} to '{client_file_path}'.")
else:
    print("Error: No data loaded.")