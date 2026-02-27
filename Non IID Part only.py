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

def load_data(file_paths):
    df_list = []
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if not df.empty:
                    df_list.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return pd.concat(df_list, ignore_index=True) if df_list else None

def non_iid_partition_with_sender_skew(df, num_clients, sender_distribution):
    partitions = []
    senders = df['sender'].unique()

    for client in range(num_clients):
        client_data = pd.DataFrame()
        for sender in senders:
            sender_data = df[df['sender'] == sender]
            n_samples = min(sender_distribution.get(sender, 0), len(sender_data))
            if n_samples > 0:
                sampled_data = sender_data.sample(n=n_samples, random_state=client)
                client_data = pd.concat([client_data, sampled_data])

        if client_data.empty:
            raise ValueError(f"No data available for client {client + 1}. Ensure sufficient samples in the dataset.")

        partitions.append(client_data.sample(frac=1).reset_index(drop=True))

    return partitions

# Main Execution
data = load_data(file_paths)

if data is not None:
    num_clients = 5  # Set the number of clients
    sender_distribution = {
        9: 200,
        15: 300,
        21: 70,
        27: 100,
        33: 50,
        38: 20
    }

    client_data = non_iid_partition_with_sender_skew(data, num_clients, sender_distribution)

    for client in range(num_clients):
        client_file_path = f'client_data_non_iid_{client + 1}.csv'
        client_data[client].to_csv(client_file_path, index=False)
        print(f"Saved data for client {client + 1} to '{client_file_path}'.")
else:
    print("Error: No data loaded.")