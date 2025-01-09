from mpi4py import MPI
import numpy as np
from scipy.special import expit
import pandas as pd


def logistic_cd_base_mpi(
    X_local,        # Local partition of X, shape (n_local, p)
    y_local,        # Local partition of y, shape (n_local,)
    beta,           # Global coefficient vector, shape (p,) or (p,1)
    beta_0,         # Global intercept (float)
    max_iter=2000,
    tol=1e-4,
    lambda_para=0.2,
    alpha_para=0.5
):
    """
    MPI-Parallel Logistic Coordinate Descent with s-step block updates (Elastic Net).
    Each rank contains a portion of the rows of X, y.
    """
    s = 1
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Flatten beta if necessary
    if beta.ndim == 2 and beta.shape[1] == 1:
        beta = beta.flatten()

    n_local, p = X_local.shape
    n = comm.allreduce(n_local, op=MPI.SUM)  # total #samples across all ranks
    epsilon = 1e-8

    epoch = 0
    beta_old = beta.copy()

    # # of block-steps per epoch
    updates_per_epoch = int(np.ceil(p / s))
    # total number of block updates ( = max_iter epochs * block-steps/epoch )
    total_iter = max_iter * updates_per_epoch

    # ------------------------------------------------------------------
    # Helper: logistic loss w/ elastic net
    # ------------------------------------------------------------------
    def logistic_loss(beta_0_, beta_, X_loc, y_loc):
        # local negative log-likelihood
        sig_loc = expit(beta_0_ + X_loc @ beta_)
        # avoid log(0)
        eps = 1e-8
        local_ll = -(
            y_loc @ np.log(sig_loc + eps) + 
            (1 - y_loc) @ np.log(1 - sig_loc + eps)
        )
        return local_ll  # partial sum

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    #s = 1
    for iteration in range(total_iter):
        # Determine block of coordinates to update
        start = (iteration * s) % p
        end = start + s
        if end <= p:
            sampled_indices = np.arange(start, end)
        else:
            # Wrap around if needed
            sampled_indices = np.concatenate([np.arange(start, p), np.arange(0, end % p)])

        # Check if we've completed one full pass => an "epoch"
        if (iteration % updates_per_epoch == 0) and (iteration != 0):
            epoch += 1

            # -------- Evaluate progress (logistic loss + penalty) --------
            # Local partial negative log-likelihood
            local_ll = logistic_loss(beta_0, beta, X_local, y_local)
            # sum across ranks
            global_ll = comm.allreduce(local_ll, op=MPI.SUM)
            # scale by 1/n
            global_ll /= n

            # Add elastic net penalty
            l2_term = 0.5 * np.sum(beta**2)
            l1_term = np.sum(np.abs(beta))
            penalty = lambda_para * ((1 - alpha_para) * l2_term + alpha_para * l1_term)
            loss_val = global_ll + penalty

            # diff_norm
            diff_local_sq = np.sum((beta_old - beta)**2)
            diff_global_sq = comm.allreduce(diff_local_sq, op=MPI.SUM)
            diff_norm = np.sqrt(diff_global_sq)

            if rank == 0:
                print(f"Epoch {epoch}, Loss={loss_val:.6f}, diff_beta={diff_norm:.6f}")

            # Convergence check
            converged = (diff_norm < tol * np.sqrt(p))
            converged_flag = comm.bcast(converged, root=0)
            if converged_flag:
                if rank == 0:
                    print(f"Converged at epoch {epoch}")
                break

            beta_old = beta.copy()

        # ------------------------------------------------------------------
        # Update step for the current block
        # ------------------------------------------------------------------
        sig_local = expit(beta_0 + X_local @ beta)  # shape (n_local,)
        W_local = sig_local * (1 - sig_local)       # shape (n_local,)

        # Intercept update
        sum_y_local   = np.sum(y_local) 
        sum_sig_local = np.sum(sig_local)
        sum_W_local   = np.sum(W_local)
        sum_y = comm.allreduce(sum_y_local, op=MPI.SUM)
        sum_sig = comm.allreduce(sum_sig_local, op=MPI.SUM)
        sum_W = comm.allreduce(sum_W_local, op=MPI.SUM)

        delta_beta_0 = (sum_y - sum_sig) / (sum_W + epsilon)
        beta_0 += delta_beta_0

        # For block updates, we need:
        #   A[l] = sum_i X[i,l] * sig[i]
        #   XY[l] = sum_i X[i,l] * y[i]
        # and
        #   B[l,m] = sum_i X[i,l] * W[i] * X[i,m]
        X_1s_local = X_local[:, sampled_indices]  # shape (n_local, s)

        # Local partial sums
        A_local  = X_1s_local.T @ sig_local                      # shape (s,)
        B_local  = X_1s_local.T @ (W_local[:, None] * X_1s_local) # shape (s, s)

        # Also gather y@X_1s[:, l] for the same block
        # => local_xy[l] = sum_i y[i] * X[i, sampled_indices[l]]
        # We'll build an array shape (s,) for block
        local_xy = np.array([y_local @ X_1s_local[:, l] for l in range(X_1s_local.shape[1])])

        # Now Allreduce to get the global sums
        A_global  = np.zeros_like(A_local)
        B_global  = np.zeros_like(B_local)
        xy_global = np.zeros_like(local_xy)

        comm.Allreduce([A_local, MPI.DOUBLE], [A_global, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([B_local, MPI.DOUBLE], [B_global, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([local_xy, MPI.DOUBLE], [xy_global, MPI.DOUBLE], op=MPI.SUM)

        # Now each rank can do the same coordinate updates using the global sums
        for l in range(len(sampled_indices)):
            idx = sampled_indices[l]

            B_ll = B_global[l, l]  # sum_i X[i,l]^2 * W[i]
            A_l  = A_global[l]     # sum_i X[i,l] * sig[i]
            XY_l = xy_global[l]    # sum_i X[i,l] * y[i]

            # summation_term from previously updated coords in the same block:
            block_indices  = sampled_indices[:l]
            summation_term = B_global[l, :l] @ (beta[block_indices] - beta_old[block_indices])

            # numerator = (1/n)*[ (y@X_l) - A_l - summation_term ] - lambda*(1-alpha)*beta_l
            numerator = (1.0 / n)*(XY_l - A_l - summation_term) \
                        - lambda_para * (1 - alpha_para) * beta[idx]
            denominator = (1.0 / n)*B_ll + lambda_para*(1 - alpha_para) + epsilon

            db_l   = numerator / denominator
            l1_val = lambda_para*alpha_para / denominator

            candidate = beta[idx] + db_l
            if abs(candidate) <= l1_val:
                beta[idx] = 0.0
            elif candidate < 0:
                beta[idx] = candidate + l1_val
            else:
                beta[idx] = candidate - l1_val

    # Reshape if you want (p,1)
    return beta.reshape(-1, 1), beta_0
