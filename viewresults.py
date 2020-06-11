import click
import csv
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def print_final_weights():
    w = np.load('log/weight.npy', allow_pickle=True)
    print('weights:')
    k=0
    for i in range(len(w)):
        print(f'\tNode {k}: {w[k]}')
        k+=1

def plot_local_results(n_nodes, local):
    plt.figure(f'Local Results, {n_nodes} nodes')
    plt.subplot(121)
    plt.title('Gap')
    plt.xlabel('global iteration step')
    plt.ylabel('$\log_{10} (local\ gap)$')
    for i in range(n_nodes):
        plt.plot(local[i]['i_iter'], np.log10(local[i]['local_gap']), label=f'node {i}')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.subplot(122)
    plt.title('Time per iteration')
    plt.xlabel('global iteration step')
    plt.ylabel('local time')
    for i in range(n_nodes):
        plt.plot(local[i]['i_iter'], local[i]['time'], label=f'node {i}')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.show()

def plot_duality_gap(n_nodes, data):
    # Plot primal and dual
    plt.figure(f'Global Results, {n_nodes} nodes')
    plt.title('Primal-Dual Gap')
    plt.xlabel('global iteration step')
    plt.ylabel('global gap')
    plt.stackplot(data['i_iter'], data['P'], data['D'], labels=['Primal', 'Dual'])
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.show()

def plot_primal_dual(n_nodes, data):
    plt.figure(f'Primal and Dual Values, {n_nodes} nodes')
    plt.subplot(121)
    plt.title('Primal')
    plt.xlabel('global iteration step')
    plt.ylabel('$\mathcal{P}(w)$')
    plt.plot(data['i_iter'], data['P'])
    
    plt.subplot(122)
    plt.title('Dual')
    plt.xlabel('global iteration step')
    plt.ylabel('$\mathcal{D}(x)$')
    plt.plot(data['i_iter'], data['D'])

    plt.show()

def plot_minimizers(n_nodes, data):
    plt.figure(f'Minimizer Values, {n_nodes} nodes')
    plt.title('Minimizer functions per iteration')
    plt.subplot(221)
    plt.xlabel('global iteration step')
    plt.ylabel('$f(Ax)$')
    plt.plot(data['i_iter'], data['f'])
    
    plt.subplot(222)
    plt.xlabel('global iteration step')
    plt.ylabel('$g(x)$')
    plt.plot(data['i_iter'], data['g'])

    plt.subplot(223)
    plt.xlabel('global iteration step')
    plt.ylabel('$f^*(w)$')
    plt.plot(data['i_iter'], data['f_conj'])
    
    plt.subplot(224)
    plt.xlabel('global iteration step')
    plt.ylabel('$g^*(-w^TA)$')
    plt.plot(data['i_iter'], data['g_conj'])

    plt.show()

@click.command()
@click.option('--n_nodes', type=click.INT, help='Number of mpi processes')
def plot_results(n_nodes):
    r"""
    Plots
    1. Absolute CV2
    2. Relative CV2

    """
    local_results = [pd.read_csv(f'log/{i}result.csv').loc[1:] for i in range(n_nodes)]
    global_results = pd.read_csv('log/result.csv').loc[1:]
    
    plot_local_results(n_nodes, local_results)
    plot_minimizers(n_nodes, global_results)
    plot_primal_dual(n_nodes, global_results)
    plot_duality_gap(n_nodes, global_results)
    
if __name__ == '__main__':
    plot_results()