from mpi4py import MPI
import numpy as np
import time
import pandas as pd
from logistic_cd_base import logistic_cd_base_mpi  # Import base function
from s_step_logistic_cd_mpi import logistic_cd_fast_mpi  # Import s-step variant
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
        file_path = "/Users/shaozishan/Desktop/Research/coordinate descent/data/colon-cancer.txt"
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

    # Run parallel logistic coordinate descent (s-step variant)
    s = 1024
    max_iter = 2000
    start_time_s_step = time.time()
    beta_s_step, beta_0_s_step = logistic_cd_fast_mpi(X_train, y_train, beta_init, beta_0_init, s, max_iter, tol=1e-4, lambda_para=0.2, alpha_para=0.5)
    total_time_s_step = time.time() - start_time_s_step

    # Run parallel logistic coordinate descent (base variant)
    start_time_base = time.time()
    beta_base, beta_0_base = logistic_cd_base_mpi(X_train, y_train, beta_init, beta_0_init, max_iter=max_iter, tol=1e-4, lambda_para=0.2, alpha_para=0.5)
    total_time_base = time.time() - start_time_base

    # Only rank 0 evaluates and prints results
    if rank == 0:
        # Evaluate on test set for s-step variant
        logits_test_s_step = beta_0_s_step + X_test @ beta_s_step
        y_prob_s_step = expit(logits_test_s_step).ravel()
        y_pred_s_step = (y_prob_s_step >= 0.5).astype(int)
        accuracy_s_step = accuracy_score(y_test, y_pred_s_step)

        # Evaluate on test set for base variant
        logits_test_base = beta_0_base + X_test @ beta_base
        y_prob_base = expit(logits_test_base).ravel()
        y_pred_base = (y_prob_base >= 0.5).astype(int)
        accuracy_base = accuracy_score(y_test, y_pred_base)

        # Print performance and comparison
        print(f"\nCustom Logistic CD (s-step) => Accuracy: {accuracy_s_step:.4f}, Time: {total_time_s_step:.2f}s")
        print("First 10 elements of beta (s-step):", beta_s_step[:10].ravel())
        print("Final beta_0 (s-step):", beta_0_s_step)

        print(f"\nCustom Logistic CD (base) => Accuracy: {accuracy_base:.4f}, Time: {total_time_base:.2f}s")
        print("First 10 elements of beta (base):", beta_base[:10].ravel())
        print("Final beta_0 (base):", beta_0_base)

        # Comparison summary
        print("\nComparison Summary:")
        print(f"Accuracy Difference: {accuracy_base - accuracy_s_step:.4f}")
        print(f"Time Difference: {total_time_base - total_time_s_step:.2f}s")

if __name__ == "__main__":
    main()
