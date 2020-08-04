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

    def __init__(self, solver, output_dir, ckpt_freq, graph, exit_time=None, split_by='samples', mode='local', alg='cola', Ak=None, Ak_test=None, y_test=None, y_offset=0.0, verbose=1):
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
        self.Ak = Ak
        self.Ak_test = Ak_test
        self.y_test = y_test
        self.do_prediction_tests = self.Ak_test is not None and self.y_test is not None
        self.y_offset = y_offset

        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.graph = graph

        self.alg = alg

        self.solver = solver

        self.running_time = 0
        self.previous_time = time.time()
        self.exit_time = exit_time or np.inf

        self.verbose = verbose

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

        self._sigma_sum = None

    # def _all_reduce_scalar(self, scalar, op):
    #     tensor = torch.DoubleTensor([scalar])
    #     comm.all_reduce(tensor, op=op)
    #     return float(tensor[0])

    # def _all_reduce_tensor(self, tensor, op):
    #     t = tensor.clone()
    #     comm.all_reduce(t, op=op)
    #     return t

    def update_sigma_sum(self):
        sigma = np.linalg.norm(self.Ak.todense()) ** 2
        n = self.Ak.shape[1]
        partial = n**2 * sigma
        self._sigma_sum = comm.all_reduce(partial, 'SUM')

    @property
    def sigma_sum(self):
        if self._sigma_sum is None:
            self.update_sigma_sum()

        return self._sigma_sum
            

    def log(self, vk, Akxk, xk, i_iter, solver, Ak_scale=1, delta_xk=None, intercept=0.0, cert_cv=0.0):
        # Skip the time for logging
        self.running_time += time.time() - self.previous_time

        if self.mode == 'local':
            self._log_local(vk, Akxk, xk, i_iter, solver, delta_xk, cert_cv=cert_cv)
        elif self.mode == 'global':
            self._log_global(vk, Akxk, xk, i_iter, solver, Ak_scale)
        elif self.mode == None:
            pass
        elif self.mode == 'all':
            self.records = self.records_l
            self._log_local(vk, Akxk, xk, i_iter, solver, delta_xk, cert_cv=cert_cv)
            self.records_l = self.records
            self.records = self.records_g
            self._log_global(vk, Akxk, xk, i_iter, solver, Ak_scale=Ak_scale, intercept=intercept)
            self.records_g = self.records
            if self.verbose >= 2:
                print(f"[{comm.get_rank()}] Certificate, Iter {self.records[-1]['i_iter']}: "
                    f"global_gap={self.records[-1]['gap']:10.5e}; "
                    f"local_gap={self.records_l[-1]['cert_gap']:10.5e}, local_cv={self.records_l[-1]['cv2']:10.5e}")
        else:
            raise NotImplementedError("[local, global, all, None] are expected mode, got {}".format(self.mode))

        self.previous_time = time.time()

        max_running_time = comm.all_reduce(self.running_time, op='MAX')
        gap = 100
        if self.mode == 'all':
            gap = self.records_g[-1]['gap']
        return max_running_time > self.exit_time or abs(gap) < 1e-6

    def _log_local(self, vk, Akxk, xk, i_iter, solver, delta_xk=None, cert_cv=0.0):
        record = {}
        record['i_iter'] = i_iter
        record['time'] = self.running_time
        try:
            record['local_gap'] = self.solver.solver.gap_
            record['n_iter_'] = self.solver.solver.n_iter_
        except:
            record['local_gap'] = 0
            record['n_iter_'] = 0

        record['delta_xk'] = norm(delta_xk) if delta_xk is not None else np.nan

        record['cert_gap'] = (vk @ solver.grad_f(vk) + solver.gk(xk) + solver.gk_conj(solver.grad_f(vk)))
        record['cert_cv'] = cert_cv

        gap_rhs = 1 / (2 * comm.get_world_size())
        cv_rhs = (1 - self.graph.beta) / (2 * np.sqrt(comm.get_world_size()) * np.sqrt(self.sigma_sum))

        record['cert_gap_scaled'] = record['cert_gap'] / gap_rhs
        record['cert_cv_scaled'] = record['cert_cv'] / cv_rhs

        self.records.append(record)

        if self.verbose >= 2:
            print("Iter {i_iter:5}, Time {time:10.5e}: delta_xk={delta_xk:10.5e}, local_gap={local_gap:10.5e}, local_iters {n_iter_}".format(**record))

    def _log_global(self, vk, Akxk, xk, i_iter, solver, Ak_scale=1, intercept=0.0):
        record = {}
        record['i_iter'] = i_iter
        record['time'] = self.running_time

        # v := A x
        v = vk
        if self.alg == 'cola':
            v = comm.all_reduce(np.array(Akxk), op='SUM')
        w = self.solver.grad_f(v)

        record['res'] = norm(v - self.solver.y) / norm(self.solver.y)

        # Compute squared norm of consensus violation
        record['cv2'] = norm(vk - v, 2) ** 2
        if self.mode == 'all':
            self.records_l[-1]['cv2'] = record['cv2']

        # Compute the value of minimizer objective
        val_gk = self.solver.gk(xk)
        record['g'] = comm.all_reduce(val_gk, 'SUM')
        record['f'] = self.solver.f(v)

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

        # The primal should be monotonically decreasing
        record['P'] = record['f'] + record['g']
        record['D'] = record['f_conj'] + record['g_conj']

        # Duality gap of the global problem
        record['gap'] = (record['D'] + record['P'])
        if record['D'] > 0:
            record['gap_rel'] = record['gap'] / record['D']
        elif record['P'] > 0:
            record['gap_rel'] = record['gap'] / record['P']


        if self.do_prediction_tests:
            coef = xk/Ak_scale
            y_predict = self.Ak_test*coef - intercept 
            y_predict = comm.all_reduce(np.asarray(y_predict), op='SUM') 
            y_predict += self.y_offset
            y_test_avg = np.average(self.y_test)
            record['n_train'] = self.solver.y.shape[0]
            record['n_test'] = self.y_test.shape[0]
            record['rmse'] = np.sqrt(np.average((y_predict - self.y_test)**2))
            record['r2'] = 1.0 - np.sum((y_predict - self.y_test)**2)/np.sum((self.y_test - y_test_avg)**2)
            record['max_rel'] = np.amax(np.abs(y_predict - self.y_test)/self.y_test)
            record['l2_rel'] = np.linalg.norm(self.y_test-y_predict, 2)/np.linalg.norm(self.y_test, 2)

        self.records.append(record)

        if self.rank == 0:
            if self.verbose >= 2:
                print("Iter {i_iter:5}, Time {time:10.5e}: gap={gap:10.3e}, P={P:10.3e}, D={D:10.3e}, f={f:10.3e}, "
                  "g={g:10.3e}, f_conj={f_conj:10.3e}, g_conj={g_conj:10.3e}".format(**record))
                

    def save(self, Akxk, xk, intercept=0.0, weightname=None, logname=None):
        rank = self.rank
        if logname:
            if self.mode == 'all':
                self.records = self.records_g
                logfile = os.path.join(self.output_dir, f'{rank}'+logname)
                pd.DataFrame(self.records_l).to_csv(logfile)
                if self.verbose >= 2:
                    print("Data has been save to {} on node {}".format(logfile, rank))
            if rank == 0:
                logfile = os.path.join(self.output_dir, logname)
                pd.DataFrame(self.records).to_csv(logfile)
                if self.verbose >= 2:
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
                if self.verbose >= 3:
                    print(f'node {rank} weights: {weight}')
                weight = comm.reduce(weight, root=0, op='SUM')
                intercept = comm.reduce(intercept, root=0, op='SUM')

            if rank == 0:
                weightfile = os.path.join(self.output_dir, weightname)
                weight.dump(weightfile)
                if self.verbose >= 2:
                    print("Weight has been save to {} on node 0".format(weightfile))
                intercept = self.y_offset - intercept
                if abs(intercept) > 1e-6:
                    interceptfile = os.path.join(self.output_dir, weightname.replace('weight', 'intercept'))
                    np.asarray([intercept]).dump(interceptfile)
                    if self.verbose >= 2:
                        print("Intercept ({}) has been save to {} on node 0".format(intercept, interceptfile))

    def show_test_statistics(self, xk_final, n_train, intercept=0, Ak_test=None, y_test=None):
        comm.barrier()
        if Ak_test is None:
            Ak_test = self.Ak_test
        if y_test is None:
            y_test = self.y_test
        if Ak_test is None or y_test is None:
            raise TypeError('Ak_test and y_test must not be None')

        n_test = y_test.shape[0]
        
        if self.mode in ['global', 'all']:
            rmse = self.records_g[-1]['rmse']
            r2 = self.records_g[-1]['r2']
            max_rel = self.records_g[-1]['max_rel']
            l2_rel = self.records_g[-1]['l2_rel']
        else:
            y_predict = Ak_test*xk_final - intercept
            y_predict = comm.all_reduce(np.asarray(y_predict), op='SUM') 
            y_predict += self.y_offset

            y_test_avg = np.average(y_test)
            rmse = np.sqrt(np.average((y_predict - y_test)**2))
            r2 = 1.0 - np.sum((y_predict - y_test)**2)/np.sum((y_test - y_test_avg)**2)
            max_rel = np.amax(np.abs(y_predict - y_test)/y_test)
            l2_rel = np.linalg.norm(y_test-y_predict, 2)/np.linalg.norm(y_test, 2)

        if self.verbose >= 1 and comm.get_rank() == 0:
            print(f'|-> Test Statistics ({n_train}/{n_test}/{n_train + n_test}): ')
            print(f'|---> max. rel. error = {max_rel}')
            print(f'|--->   rel. L2 error = {l2_rel}')
            print(f'|--->            RMSE = {rmse}')
            print(f'|--->             R^2 = {r2}')
