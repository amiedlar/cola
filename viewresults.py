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

def plot_local_results(n_nodes, local, x_axis='i_iter', x_label='global iteration step', comp_data=None):
    plt.figure(f'Local Results, {n_nodes} nodes')
    if comp_data is None:
        plt.subplot(111)
        plt.title('Gap')
    else:
        plt.subplot(121)
        plt.title('[COLA]Gap')
    plt.xlabel(x_label)
    plt.ylabel('$\log_{10} (local\ gap)$')
    for i in range(n_nodes):
        plt.plot(local[i][x_axis], np.log10(local[i]['local_gap']), label=f'node {i}')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    if comp_data is None:
        plt.show()
        return

    plt.subplot(122)
    plt.title('[COCOA]Gap')
    plt.xlabel(x_label)
    plt.ylabel('$\log_{10} (local\ gap)$')
    for i in range(n_nodes):
        plt.plot(comp_data[i][x_axis], np.log10(comp_data[i]['local_gap']), label=f'node {i}')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.show()

def plot_duality_gap(n_nodes, data, x_axis='i_iter', x_label='global iteration step', comp_data=None):
    # Plot primal and dual
    fig = plt.figure(f'Global Results, {n_nodes} nodes')
    gs = GridSpec(3, 2, figure=fig)

    fig.add_subplot(gs[0,0])
    plt.title('Primal')
    plt.xlabel(x_label)
    plt.ylabel('$\mathcal{P}(w)$')
    plt.plot(data[x_axis], data['P'], label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], comp_data['P'], label='CoCoA')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    fig.add_subplot(gs[0,1])
    plt.title('Dual')
    plt.xlabel(x_label)
    plt.ylabel('$\mathcal{D}(x)$')
    plt.plot(data[x_axis], data['D'], label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], comp_data['D'], label='CoCoA')

    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    fig.add_subplot(gs[1:,0:])
    plt.title('Primal-Dual Gap')
    plt.xlabel(x_label)
    plt.ylabel('$\log_{10}(global\ gap)$')
    plt.plot(data[x_axis], np.log10(np.abs(data['gap'])), label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], np.log10(np.abs(comp_data['gap'])), label='CoCoA')

    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.show()

def plot_primal_dual(n_nodes, data, x_axis='i_iter', x_label='global iteration step', comp_data=None):
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

def plot_minimizers(n_nodes, data, x_axis='i_iter', x_label='global iteration step', comp_data=None):
    plt.figure(f'Minimizer Values, {n_nodes} nodes')
    plt.title('Minimizer functions per iteration')
    plt.subplot(221)
    plt.xlabel(x_label)
    plt.ylabel('$f(Ax)$')
    plt.plot(data[x_axis], data['f'], label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], comp_data['f'], label='CoCoA')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.subplot(222)
    plt.xlabel(x_label)
    plt.ylabel('$g(x)$')
    plt.plot(data[x_axis], data['g'], label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], comp_data['g'], label='CoCoA')

    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.subplot(223)
    plt.xlabel(x_label)
    plt.ylabel('$f^*(w)$')
    plt.plot(data[x_axis], data['f_conj'], label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], comp_data['f_conj'], label='CoCoA')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.subplot(224)
    plt.xlabel(x_label)
    plt.ylabel('$g^*(-w^TA)$')
    plt.plot(data[x_axis], data['g_conj'], label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], comp_data['g_conj'], label='CoCoA')

    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.show()

@click.command()
@click.option('--n_nodes', type=click.INT, default=None, help='Number of mpi processes')
@click.option('--logdir', type=click.STRING, default='log', help='root directory of output files')
@click.option('--dataset', type=click.STRING, default='rderms', help='dataset name')
@click.option('--compare', is_flag=True)
def plot_results(n_nodes, logdir, dataset, compare):
    from os.path import join
    log_path = join(logdir, dataset) 
    comp_path = join(logdir, 'cocoa', dataset)
    local_results = []
    if not n_nodes:
        n_nodes = 1
        import os.path as path
        while True:
            next_dir = join(log_path, f'{n_nodes}')
            if path.exists(next_dir):
                break
            n_nodes+=1
            if n_nodes>256:
                Exception('No logs found.')
    
    log_path = join(log_path, f'{n_nodes}')
    comp_path = join(comp_path, f'{n_nodes}')
    assert n_nodes is not None and n_nodes>0, 'logs not found, try specifying `logdir` and `n_nodes`' 
    
    local_results = [pd.read_csv(join(log_path,f'{i}result.csv')).loc[1:] for i in range(n_nodes)]
    global_results = pd.read_csv(join(log_path, 'result.csv')).loc[1:]
    comp_local = None 
    comp_global = None 
    if compare:
        comp_local = [pd.read_csv(join(comp_path,f'{i}result.csv')).loc[1:] for i in range(n_nodes)]
        comp_global = pd.read_csv(join(comp_path, 'result.csv')).loc[1:]
    
    plot_local_results(n_nodes, local_results, x_label='Iteration Count', comp_data=comp_local)
    plot_minimizers(n_nodes, global_results, x_label='Iteration Count', comp_data=comp_global)
    plot_duality_gap(n_nodes, global_results, x_label='Iteration Count', comp_data=comp_global)
    
if __name__ == '__main__':
    plot_results()