from mpi4py import MPI
import numpy as np
import time
import pandas as pd
from s_step_logistic_cd_mpi import logistic_cd_fast_mpi  # Import your parallel function
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.special import expit



def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only rank 0 loads the data
    if rank == 0:
        # Load dataset
        #file_path = "/Users/shaozishan/Desktop/Research/coordinate descent/colon-cancer.txt"
        file_path = "/Users/shaozishan/Desktop/Research/coordinate descent/data/diabetes_scale.txt"
        X_list, y_list = [], []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = int(float(parts[0]))
                y_list.append(0 if label == -1 else label)
                features = {int(kv.split(':')[0]): float(kv.split(':')[1]) for kv in parts[1:]}
                X_list.append(features)

        # Convert to DataFrame
        X_df = pd.DataFrame(X_list).fillna(0).sort_index(axis=1)
        y_arr = np.array(y_list)
        print("Data loaded. Shape of X:", X_df.shape)

        # Train/test split
        n_samples = X_df.shape[0]
        split_index = int(n_samples * 0.8)
        X_train = X_df.iloc[:split_index].values
        y_train = y_arr[:split_index]
        X_test = X_df.iloc[split_index:].values
        y_test = y_arr[split_index:]

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    else:
        # Other ranks initialize empty placeholders
        X_train, y_train, X_test, y_test = None, None, None, None

    # Broadcast the data to all ranks
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)

    # Initialize beta and beta_0
    n, p = X_train.shape
    beta_init = np.zeros((p, 1))
    eps = 1e-12
    y_mean = np.mean(y_train)
    beta_0_init = np.log((y_mean + eps) / (1 - y_mean + eps))

    # Run parallel logistic coordinate descent
    s = 1024
    max_iter = 2000
    start_time = time.time()
    beta, beta_0 = logistic_cd_fast_mpi(X_train, y_train, beta_init, beta_0_init, s, max_iter, tol=1e-4, lambda_para=0.2, alpha_para=0.5)
    total_time = time.time() - start_time

    # Only rank 0 evaluates and prints results
    if rank == 0:
        # Evaluate on test set
        logits_test = beta_0 + X_test @ beta
        y_prob = expit(logits_test).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)

        # Print performance and results
        print(f"\nCustom Logistic CD => Accuracy: {accuracy:.4f}, Time: {total_time:.2f}s")
        print("First 10 elements of beta:", beta[:10].ravel())
        print("Final beta_0:", beta_0)

if __name__ == "__main__":
    main()
