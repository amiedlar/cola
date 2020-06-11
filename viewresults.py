import click
import csv
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

def print_final_weights():
    w = np.load('log/weight.npy', allow_pickle=True)
    print('weights:')
    k=0
    for i in range(len(w)):
        print(f'\tNode {k}: {w[k]}')
        k+=1

def plot_local_results(n_nodes, local, x_axis='i_iter', x_label='global iteration step'):
    plt.figure(f'Local Results, {n_nodes} nodes')
    plt.subplot(121)
    plt.title('Gap')
    plt.xlabel(x_label)
    plt.ylabel('$\log_{10} (local\ gap)$')
    for i in range(n_nodes):
        plt.plot(local[i][x_axis], np.log10(local[i]['local_gap']), label=f'node {i}')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.subplot(122)
    plt.title('Time per iteration')
    plt.xlabel(x_label)
    plt.ylabel('local time')
    for i in range(n_nodes):
        plt.plot(local[i][x_axis], local[i]['time'], label=f'node {i}')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.show()

def plot_duality_gap(n_nodes, data, x_axis='i_iter', x_label='global iteration step'):
    # Plot primal and dual
    fig = plt.figure(f'Global Results, {n_nodes} nodes')
    gs = GridSpec(3, 2, figure=fig)

    fig.add_subplot(gs[0,0])
    plt.title('Primal')
    plt.xlabel(x_label)
    plt.ylabel('$\mathcal{P}(w)$')
    plt.plot(data[x_axis], data['P'])
    
    fig.add_subplot(gs[0,1])
    plt.title('Dual')
    plt.xlabel(x_label)
    plt.ylabel('$\mathcal{D}(x)$')
    plt.plot(data[x_axis], data['D'])

    fig.add_subplot(gs[1:,0:])
    plt.title('Primal-Dual Gap')
    plt.xlabel(x_label)
    plt.ylabel('$\log_{10}(global\ gap)$')
    plt.plot(data[x_axis], np.log10(np.abs(data['gap'])))

    plt.show()

def plot_primal_dual(n_nodes, data, x_axis='i_iter', x_label='global iteration step'):
    plt.figure(f'Primal and Dual Values, {n_nodes} nodes')
    plt.subplot(121)
    plt.title('Primal')
    plt.xlabel(x_label)
    plt.ylabel('$\mathcal{P}(w)$')
    plt.plot(data[x_axis], data['P'])
    
    plt.subplot(122)
    plt.title('Dual')
    plt.xlabel(x_label)
    plt.ylabel('$\mathcal{D}(x)$')
    plt.plot(data[x_axis], data['D'])

    plt.show()

def plot_minimizers(n_nodes, data, x_axis='i_iter', x_label='global iteration step'):
    plt.figure(f'Minimizer Values, {n_nodes} nodes')
    plt.title('Minimizer functions per iteration')
    plt.subplot(221)
    plt.xlabel(x_label)
    plt.ylabel('$f(Ax)$')
    plt.plot(data[x_axis], data['f'])
    
    plt.subplot(222)
    plt.xlabel(x_label)
    plt.ylabel('$g(x)$')
    plt.plot(data[x_axis], data['g'])

    plt.subplot(223)
    plt.xlabel(x_label)
    plt.ylabel('$f^*(w)$')
    plt.plot(data[x_axis], data['f_conj'])
    
    plt.subplot(224)
    plt.xlabel(x_label)
    plt.ylabel('$g^*(-w^TA)$')
    plt.plot(data[x_axis], data['g_conj'])

    plt.show()

@click.command()
@click.option('--n_nodes', type=click.INT, default=None, help='Number of mpi processes')
@click.option('--logdir', type=click.STRING, default='log', help='root directory of output files')
@click.option('--dataset', type=click.STRING, default='rderms', help='dataset name')
def plot_results(n_nodes, logdir, dataset):
    from os.path import join
    log_path = join(logdir, dataset) 
    local_results = []
    if not n_nodes:
        n_nodes = 0
        import os.path as path
        while True:
            next_file = join(log_path, f'{n_nodes}result.csv')
            if path.exists(next_file):
                local_results.append(pd.read_csv(next_file).loc[1:])
                n_nodes += 1
            else:
                break
    else:
        local_results = [pd.read_csv(join(log_path,f'{i}result.csv')).loc[1:] for i in range(n_nodes)]

    global_results = pd.read_csv(join(log_path, 'result.csv')).loc[1:]
    
    plot_local_results(n_nodes, local_results, x_axis='time', x_label='Time Elapsed (s)')
    plot_minimizers(n_nodes, global_results, x_axis='time', x_label='Time Elapsed (s)')
    plot_duality_gap(n_nodes, global_results, x_axis='time', x_label='Time Elapsed (s)')
    
if __name__ == '__main__':
    plot_results()