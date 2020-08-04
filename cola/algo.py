r"""CoCoA family to use."""
import warnings
import numpy as np

from . import communication as comm


def run_algorithm(algorithm, Ak, b, solver, gamma, theta, max_global_steps, local_iters, n_nodes, graph, monitor, fit_intercept=False):
    r"""Run cocoa family algorithms."""
    with warnings.catch_warnings():
        # warnings.filterwarnings(
        #     "ignore",
        #     'Objective did not converge. You might want to increase the number of iterations. '
        #     'Fitting data with very small alpha may cause precision problems.')

        comm.barrier()
        if algorithm == 'cola':
            Akxk, xk, intercept = cola(Ak, b, solver, gamma, theta,
                            max_global_steps, local_iters, n_nodes, graph, monitor, fit_intercept)
        # elif algorithm == 'cocoa':
        #     Akxk, xk = cocoa(Ak, b, solver, gamma, theta,
        #                     max_global_steps, local_iters, n_nodes, monitor)
        else:
            raise NotImplementedError()
    return Akxk, xk, intercept


def cola(Ak, b, localsolver, gamma, theta, global_iters, local_iters, K, graph, monitor, fit_intercept=False):
    if gamma <= 0 or gamma > 1:
        raise ValueError("gamma should in (0, 1]: got {}".format(gamma))

    from .dataset import _preprocess_data
    Ak, b, Ak_offset, b_offset, Ak_scale = _preprocess_data(Ak.todense(), b, fit_intercept, normalize=True, return_mean=True)
    from scipy.sparse import csc_matrix
    Ak = csc_matrix(Ak)
    monitor.y_offset = b_offset or 0.0

    # Shape of the matrix
    n_rows, n_cols = Ak.shape

    # Current rank of the node
    rank = comm.get_rank()

    # Initialize
    xk = np.zeros(n_cols)
    Akxk = np.zeros(n_rows)

    # Keep a list of neighborhood and their estimates of v
    local_lookups = graph.get_neighborhood(rank)
    local_vs = {node_id: np.zeros(n_rows)
                for node_id, _ in local_lookups.items()}

    sigma = gamma * K
    localsolver.dist_init(Ak, b, theta, local_iters, sigma)

    # Initial
    comm.p2p_communicate_neighborhood_tensors(
        rank, local_lookups, local_vs)

    monitor.log(np.zeros(n_rows), Akxk, xk, 0, localsolver)
    for i_iter in range(1, 1 + global_iters):
        # Average the local estimates of neighborhood and self
        averaged_v = comm.local_average(n_rows, local_lookups, local_vs)

        # Solve the suproblem using this estimates
        delta_xk, delta_v = localsolver.solve(averaged_v, Akxk, xk)

        # update local variables
        xk += gamma * delta_xk
        Akxk += gamma * delta_v

        # update shared variables
        averaged_v += gamma * delta_v * K
        local_vs[rank] = averaged_v
        comm.p2p_communicate_neighborhood_tensors(
            rank, local_lookups, local_vs)

        avg_grad_f = np.zeros_like(averaged_v)
        for node_id in local_lookups:
            avg_grad_f += localsolver.grad_f(local_vs[node_id])
        avg_grad_f /= len(local_lookups)
        cert_cv = np.linalg.norm(localsolver.grad_f(averaged_v) - avg_grad_f, 2)

        _intercept = np.dot(Ak_offset, xk.T/Ak_scale) if Ak_offset is not None else 0.0 
        if monitor.log(averaged_v, Akxk, xk, i_iter, localsolver, Ak_scale=Ak_scale, delta_xk=delta_xk, intercept=_intercept, cert_cv=cert_cv):
            if monitor.verbose >= 1 and rank == 0:
                print(f'break @ iter {i_iter}.')
            break

        if (i_iter % monitor.ckpt_freq) == 0:
            monitor.save(
                Akxk, xk/Ak_scale, intercept=_intercept, weightname='weight_epoch_{}.npy'.format(i_iter))

    return Akxk, xk/Ak_scale, _intercept


from mpi4py import MPI
def cocoa(Ak, b, localsolver, gamma, theta, global_iters, local_iters, K, monitor):
    if gamma <= 0 or gamma > 1:
        raise ValueError("gamma should in (0, 1]: got {}".format(gamma))

    # Shape of the matrix
    n_rows, n_cols = Ak.shape

    # Current rank of the node
    rank = comm.get_rank()

    # Initialize
    xk = np.zeros(n_cols)
    v = np.zeros(n_rows)

    sigma = gamma * K
    localsolver.dist_init(Ak, b, theta, local_iters, sigma)

    # Initial
    monitor.log(np.zeros(n_rows), v, xk, 0, localsolver)
    for i_iter in range(1, 1 + global_iters):
        # Solve the suproblem using this estimates
        delta_xk, delta_vk = localsolver.solve(v, Ak*xk, xk)
        # update local variables
        xk += gamma * delta_xk

        # update shared variables
        delta_v = np.zeros_like(delta_vk)
        MPI.COMM_WORLD.Allreduce(delta_vk, delta_v, op=MPI.SUM)
        # assert (delta_v != old).any()
        v += gamma * delta_v

        if monitor.log(v, Ak*xk, xk, i_iter, localsolver, delta_xk):
            if monitor.verbose >= 2:
                print('break iterations here.')
            break

        if (i_iter % monitor.ckpt_freq) == 0:
            monitor.save(v, xk, weightname='weight_epoch_{}.npy'.format(i_iter))

    return v, xk
