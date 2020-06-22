import os
import time
import pandas as pd
import numpy as np
from numpy.linalg import norm
# import torch
# import torch.distributed as dist
from .cocoasolvers import CoCoASubproblemSolver
import cola.communication as comm


class Monitor(object):
    """ Supervising the training process. 

    This class is used to:
    * log the metrics during training (time, loss, etc);
    * save weight file and log files if specified;
    """

    def __init__(self, solver, output_dir, ckpt_freq, exit_time=None, split_by='samples', mode='local', alg='cola'):
        """
        Parameters
        ----------
        solver : CoCoASubproblemSolver
            a solver to be monitored.
        output_dir : str
            directory of output.
        ckpt_freq : Int
            frequency of the checkpoint.
        exit_time : float, optional
            exit if the program has been running for `exit_time`. (the default is None, which disable this criterion.)
        split_by : str, optional
            The data matrix is split by samples or features (the default is 'samples')
        mode : ['local', 'global', None], optional
             * `local` mode only logs duality gap of local solver. 
             * `global` mode logs duality gap of the whole program. It takes more time to compute.
        """
        assert isinstance(solver, CoCoASubproblemSolver)
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()

        self.alg = alg

        self.solver = solver

        self.running_time = 0
        self.previous_time = time.time()
        self.exit_time = exit_time or np.inf

        self.records = []
        self.records_l = []
        self.records_g = []
        self.mode = mode
        self.ckpt_freq = ckpt_freq
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # If a problem is split by samples, then the total number of data points is unknown
        # in a local node. As a result, we will defer the division to the logging time.
        self.split_by_samples = split_by == 'samples'

    # def _all_reduce_scalar(self, scalar, op):
    #     tensor = torch.DoubleTensor([scalar])
    #     comm.all_reduce(tensor, op=op)
    #     return float(tensor[0])

    # def _all_reduce_tensor(self, tensor, op):
    #     t = tensor.clone()
    #     comm.all_reduce(t, op=op)
    #     return t

    def log(self, vk, Akxk, xk, i_iter, solver, delta_xk=None):
        # Skip the time for logging
        self.running_time += time.time() - self.previous_time

        if self.mode == 'local':
            self._log_local(vk, Akxk, xk, i_iter, solver, delta_xk)
        elif self.mode == 'global':
            self._log_global(vk, Akxk, xk, i_iter, solver)
        elif self.mode == None:
            pass
        elif self.mode == 'all':
            self.records = self.records_l
            self._log_local(vk, Akxk, xk, i_iter, solver, delta_xk)
            self.records_l = self.records
            self.records = self.records_g
            self._log_global(vk, Akxk, xk, i_iter, solver)
            self.records_g = self.records
            print(f"[{comm.get_rank()}] Certificate, Iter {self.records[-1]['i_iter']}: "
                  f"global_gap={self.records[-1]['gap']:10.5e}; "
                  f"local_gap={self.records_l[-1]['cert_gap']:10.5e}, local_cv={self.records_l[-1]['cert_cv']:10.5e}")
        else:
            raise NotImplementedError("[local, global, all, None] are expected mode, got {}".format(self.mode))

        self.previous_time = time.time()

        max_running_time = comm.all_reduce(self.running_time, op='MAX')
        gap = 100
        if self.mode == 'all':
            gap = self.records_g[-1]['gap']
        return max_running_time > self.exit_time or abs(gap) < 1e-15

    def _log_local(self, vk, Akxk, xk, i_iter, solver, delta_xk=None):
        record = {}
        record['i_iter'] = i_iter
        record['time'] = self.running_time
        try:
            record['local_gap'] = self.solver.solver.gap_
            record['n_iter_'] = self.solver.solver.n_iter_
        except:
            record['local_gap'] = 0
            record['n_iter_'] = 0
        K = comm.get_world_size()
        wk = self.solver.grad_f(Akxk)
        record['delta_xk'] = delta_xk if delta_xk is not None else np.nan
        record['cert_gap'] = Akxk @ wk
        L = 1/self.solver.theta
        record['cert_cv'] = norm(wk - self.solver.grad_f(vk), 2) 
        self.records.append(record)

        print("Iter {i_iter:5}, Time {time:10.5e}: delta_xk={delta_xk:10.5e}, local_gap={local_gap:10.5e}, local_iters {n_iter_}".format(**record))

    def _log_global(self, vk, Akxk, xk, i_iter, solver):
        record = {}
        record['i_iter'] = i_iter
        record['time'] = self.running_time

        # v := A x
        v = vk
        if self.alg == 'cola':
            v = comm.all_reduce(np.array(Akxk), op='SUM')
        w = self.solver.grad_f(v)

        record['res'] = float(norm(v - self.solver.y)/norm(self.solver.y))
        val_res2 = (-w @ self.solver.Ak) @ xk
        record['res2'] = float(comm.all_reduce(val_res2, 'SUM'))
        record['wy'] = float(w @ self.solver.y)
        # Compute squared norm of consensus violation
        record['cv2'] = float(norm(vk - v, 2) ** 2)
        # Compute the value of minimizer objective
        record['mag_xk'] = norm(xk,2)
        record['mag_v'] = float(norm(v,2) ** 2)
        val_gk = self.solver.gk(xk)
        record['g'] = comm.all_reduce(val_gk, 'SUM')
        record['f'] = self.solver.f(v)
        record['f*'] = 0.5*float(norm(v,2)**2 - norm(self.solver.y,2)**2) 
        # Compute the value of conjugate objective
        val_gk_conj = self.solver.gk_conj(w)
        record['f_conj'] = self.solver.f_conj(w)
        record['g_conj'] = comm.all_reduce(val_gk_conj, op='SUM')

        if self.split_by_samples:
            n_samples = comm.all_reduce(len(solver.y), op='SUM')
        else:
            n_samples = len(solver.y)

        record['g'] /= n_samples
        record['g_conj'] /= n_samples
        record['f'] /= n_samples
        record['f_conj'] /= n_samples
        record['wy'] /= n_samples
        record['f*'] /= n_samples
        # The dual should be monotonically decreasing
        record['D'] = record['f'] + record['g']
        record['P'] = record['f_conj'] + record['g_conj']

        # Duality gap of the gloabl problem
        record['gap'] = record['D'] + record['P']

        self.records.append(record)

        if self.rank == 0:
            print('Iter {i_iter:5} DEBUG: res={res2:10.3e}, w*y={wy:10.3e}, f*={f*:10.3e}'.format(**record)) 
            print("Iter {i_iter:5}, Time {time:10.5e}: gap={gap:10.3e}, P={P:10.3e}, D={D:10.3e}, f={f:10.3e}, "
                  "g={g:10.3e}, f_conj={f_conj:10.3e}, g_conj={g_conj:10.3e}".format(**record))

    def save(self, Akxk, xk, weightname=None, logname=None):
        rank = self.rank
        if logname:
            if self.mode == 'all':
                self.records = self.records_g
                logfile = os.path.join(self.output_dir, f'{rank}'+logname)
                pd.DataFrame(self.records_l).to_csv(logfile)
                print("Data has been save to {} on node {}".format(logfile, rank))
            if rank == 0:
                logfile = os.path.join(self.output_dir, logname)
                pd.DataFrame(self.records).to_csv(logfile)
                print("Data has been save to {} on node 0".format(logfile))

        if weightname:
            if self.split_by_samples:
                Akxk = comm.reduce(Akxk, root=0, op='SUM')
                weight = Akxk

            else:
                # If features are split, then concatenate xk's weight
                size = np.array([0] * self.world_size)
                size[rank] = len(xk)
                size = comm.all_reduce(size, op='SUM')
                # the size is [len(x_0), len(x_1), ..., len(x_{K-1})]

                weight = np.zeros(sum(size))
                weight[sum(size[:rank]): sum(size[:rank]) + len(xk)] = np.array(xk)
                # print(f'node {rank} weights: {weight}')
                weight = comm.reduce(weight, root=0, op='SUM')

            if rank == 0:
                weightfile = os.path.join(self.output_dir, weightname)
                weight.dump(weightfile)
                print("Weight has been save to {} on node 0".format(weightfile))
