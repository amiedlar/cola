import click
import csv
from math import ceil
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import os
import logging
import cola.communication as comm
from mpi4py import MPI
import glob


def rgb2color(r, g, b):
    return r/255, g/255, b/255

def plot_train_test(ax, monitor, index, index_test):
    y = monitor.solver.y + monitor.model.b_offset_
    y_test = monitor.y_test
    t = np.linspace(0, 24, len(y)+len(y_test))

    ax.scatter(t[index], y, color='tab:orange', s=16, marker='x', label=f'Train Data')
    ax.scatter(t[index_test], y_test, color='tab:blue', s=16, marker='o', label=f'Test Data')
    return ax

def get_regression(monitor, index, index_test):
    Ak = monitor.Ak
    Ak_test = monitor.Ak_test

    ind = np.concatenate((index, index_test))
    rev_ind = np.asarray([-1]*len(ind), dtype=index.dtype)
    for i in range(len(ind)):
        rev_ind[ind[i]] = i

    A = np.concatenate((Ak.todense(), Ak_test.todense()))
    A = A[rev_ind, :]
    regression = monitor.model.predict(A)
    return regression

def get_dataframe(monitor, local=False):
    data = monitor.records_l[1:] if local else monitor.records_g[1:]
    return pd.DataFrame(data)

ls = ['-', '--', ':']
colors = ['tab:pink', 'tab:green', 'tab:purple']

def clean_plots():
    if comm.get_rank() == 0:
        savedir = os.path.join('out','report','img')
        if os.path.exists(savedir):
            for img in glob.glob(os.path.join(savedir, '*.png')):
                os.remove(img)

def make_intercept_plots(expname, default, center, affine, index, index_test, ):
    savedir = os.path.join('out','report','img')
    os.makedirs(savedir, exist_ok=True)
    rank = comm.get_rank()

    if rank == 0:
        # Set rc parameters
        plt.rc('font', size=11)
        plt.rc('xtick', labelsize=9)
        plt.rc('ytick', labelsize=9)
        plt.rc('lines', lw=1.4)
        plt.rc('figure', figsize=(6.5, 3.5))
        plt.rc('legend', fancybox=False, loc='upper right', fontsize='small', borderaxespad=0)
        plt.tick_params(which='major', labelsize='small')
        from matplotlib import rcsetup
        nipy = plt.cm.get_cmap(name='nipy_spectral')
        idx = 1 - np.linspace(0, 1, 20)
        plt.rc('axes', prop_cycle = rcsetup.cycler('color', nipy(idx)))

    make_intercept_regression_plot(expname, default, center, affine, index, index_test, savedir)
    make_stop_plot(expname, default, center, affine, savedir)
    # make_intercept_local_cert_plot(expname, default, center, savedir, type='gap')
    # make_intercept_local_cert_plot(expname, default, center, savedir, type='cv')
    make_intercept_global_local_plot(expname, default, center, savedir)
    make_intercept_global_local_plot(expname, default, center, savedir, no_reg=True)

def make_intercept_regression_plot(expname, default, center, affine, index, index_test, savedir):
    rank = comm.get_rank()
    
    if center is not None:
        comm.resize(center.world_size)
        default_reg = get_regression(default, index, index_test)
        center_reg = get_regression(center, index, index_test)
        comm.reset()
    if affine is not None:
        affine_reg = get_regression(affine, index, index_test)
    
    if rank == 0:
        # Regression Comp Plot
        fig, ax = plt.subplots(1,1)
        ax.set_ylabel('Meter Voltage (V)')
        plot_train_test(ax, center, index, index_test)
        t = np.linspace(0, 24, len(index)+len(index_test))
        ax.plot(t, default_reg.T, label=f'Regression - Default', linestyle='--', color='tab:purple')
        ax.plot(t, center_reg.T, label=f'Regression - Center', linestyle='-', color='black')
        if affine is not None:
            ax.plot(t, affine_reg.T, label=f'Regression - Affine', linestyle=':', color='tab:green')
        ax.legend()
        ax.set_xlabel('Time (h)')
        fig.tight_layout()
        fig.savefig(os.path.join(savedir, f'{expname}regression.png'), dpi=300)
        plt.close(fig)
    # duality gap plot

import matplotlib._color_data as mcd

def make_stop_plot(expname, default, center, affine, savedir):
    rank = comm.get_rank()
    for mon in [default, center]:
        comm.barrier()
        if mon is None:
            continue

        size = mon.world_size
        comm.resize(size)

        global_data = get_dataframe(mon)

        local_data = get_dataframe(mon, local=True)
        sendbuf = np.array(local_data['delta_xk'])
        local_updates = None
        if rank == 0:
            local_updates = np.empty([size, len(sendbuf)])
        comm.comm.Gather(sendbuf, local_updates, root=0)

        if rank == 0:
            fig, ax_l = plt.subplots(1, 1)
            fig.set_size_inches(6.5, 3.5)
            ax_r = plt.twinx(ax=ax_l)
            ax_l.set_xlabel('Iterations')
            ax_l.set_ylabel(r'$\|\|\Delta x_{k}\|\|$')
            ax_r.set_ylabel(r'$f(Ax)$')

            iters = np.asarray(local_data['i_iter'])
            for k in range(size):
                ax_l.semilogy(iters, local_updates[k,:], linestyle='--', label=r'$\|\|\Delta x_{k}\|\|$')
            ax_r.plot('i_iter', 'f', '', data=global_data, color='black', label='$f(Ax)$')
            
            ymin, ymax = ax_r.get_ylim()
            dist = ymax - ymin
            if dist < 0.1:
                ymax += (0.1 - dist)/2
                ymin -= (0.1 - dist)/2
                ax_r.set_ylim(ymin, ymax)
            
            fig.tight_layout()
            fig.savefig(os.path.join(savedir, f'{expname}{mon.name}_stop.png'), dpi=300)
            plt.close(fig)
        comm.reset()

def make_intercept_local_cert_plot(expname, default, center, savedir, type='gap'):
    rank = comm.get_rank()
    for mon in [default, center]:
        comm.barrier()
        if mon is None:
            continue

        size = mon.world_size
        comm.resize(size)

        global_data = get_dataframe(mon)

        local_data = get_dataframe(mon, local=True)
        sendbuf = np.abs(np.array(local_data[f'cert_{type}']))
        local_updates = None
        if rank == 0:
            local_updates = np.empty([size, len(sendbuf)])
        comm.comm.Gather(sendbuf, local_updates, root=0)

        if rank == 0:
            fig, ax_l = plt.subplots(1, 1)
            fig.set_size_inches(6.5, 3.5)
            ax_r = plt.twinx(ax=ax_l)
            ax_l.set_xlabel('Iterations')
            label = 'Local Gap' if type=='gap' else 'Local CV'
            ax_l.set_ylabel(label)
            ax_r.set_ylabel(r'$f(Ax)$')
            iters = np.asarray(local_data['i_iter'])
            for k in range(size):
                ax_l.semilogy(iters, local_updates[k,:], linestyle='--', label=label)
            data = np.abs(global_data['gap'])
            ax_r.semilogy('i_iter', data, '', data=global_data, color='black', label='$f(Ax)$')
            
            fig.tight_layout()
            fig.savefig(os.path.join(savedir, f'{expname}{mon.name}_cert_{type}.png'), dpi=300)
            plt.close(fig)
        comm.reset()

def make_intercept_global_local_plot(expname, default, center, savedir, no_reg=False):
    rank = comm.get_rank()
    for mon in [default, center]:
        comm.barrier()
        if mon is None:
            continue

        size = mon.world_size
        comm.resize(size)

        if rank == 0:
            global_data = get_dataframe(mon)

        local_data = get_dataframe(mon, local=True)
        local_data.replace(np.nan, 0)
        if no_reg:
            local_subproblem = np.array(local_data['subproblem']-local_data['gk'])
        else:
            local_subproblem = np.array(local_data['subproblem'])
        
        comm.reduce(local_subproblem, op='SUM', root=0)

        if rank == 0:
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(3.5, 3)
            ax.set_xlabel('Iterations')

            iters = np.asarray(local_data['i_iter'])
            label = r"$\sum_k \Gamma_{k}^{\sigma'}" + (r'-g_{[k]}$' if no_reg  else "$")
            ax.semilogy(iters, local_subproblem, color='tab:cyan', linestyle='--', label=label)

            y_axis = 'f' if no_reg else 'P'
            label = r"$f(Ax)$" if no_reg else r"$\mathcal{O}_A(x)$"
            ax.semilogy('i_iter', y_axis, '', data=global_data, color='tab:orange', label=label)

            ax.legend(loc='best')
            fig.tight_layout()

            suf = '_no_reg' if no_reg else ''
            fig.savefig(os.path.join(savedir, f'{expname}{mon.name}_thm1{suf}.png'), dpi=300)
            plt.close(fig)
        comm.reset()



