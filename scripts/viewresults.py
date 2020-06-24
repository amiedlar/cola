import click
import csv
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import os

def print_weights(dataset, k, alg='cola', logpath='log', mode='final'):
    if alg == 'cola':
        logpath = os.path.join(logpath, dataset, f'{k}')
    else:
        logpath = os.path.join(logpath, alg, dataset, k)
    if mode == 'final':
        logpath = os.path.join(logpath, 'weight.py')
    else:
        logpath = os.path.join(logpath, f'{mode}weight.py')
    w = np.load(logpath, allow_pickle=True)
    print('weights:')
    k=0
    for i in range(len(w)):
        print(f'\tNode {k}: {w[k]}')
        k+=1

def plot_residual(k, res, comp_res=None):
    
    fig = plt.figure(f'Residuals, {k} nodes')
   
    plt.title('Residuals')
    plt.xlabel('Iteration Count')
    plt.ylabel(r'$\log_{10} (\|\|x_k - x^*\|\|/\|\|x^*\|\|)$')
    plt.plot(res['i_iter'], np.log10(res['res']), label='CoLA')
    if comp_res is not None:
        plt.plot(comp_res['i_iter'], np.log10(comp_res['res']), label='CoCoA')

    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    fig.tight_layout(pad=2.)
    return fig
    

def plot_local_results(k, local, x_axis='i_iter', x_label='global iteration step', comp_data=None):
    fig = plt.figure(f'Local Results, {k} nodes')
    gs = GridSpec(3, 2, figure=fig)

    if comp_data is None:
        fig.add_subplot(gs[0,:])
        plt.title('Gap')
    else:
        fig.add_subplot(gs[0,0])
        plt.title('[COLA]Gap')
    plt.xlabel(x_label)
    plt.ylabel(r'$\log_{10} (local\ gap)$')
    for (i, node_data) in enumerate(local):
        plt.plot(node_data[x_axis], np.log10(node_data['local_gap']), label=f'node {i}')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    fig.add_subplot(gs[1:,:])
    plt.xlabel(x_label)
    plt.ylabel(r'$\|\| \Delta x_k \|\|$')
    for (i, node_data) in enumerate(local):
        plt.plot(node_data[x_axis], np.log10(node_data['delta_xk']), label=f'node {i}')

    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    if comp_data is None:
        fig.tight_layout(pad=2.)
        plt.show()
        return fig

    fig.add_subplot(gs[0,1])
    plt.title('[COCOA]Gap')
    plt.xlabel(x_label)
    plt.ylabel(r'$\log_{10} (local\ gap)$')
    for i,node_data in enumerate(comp_data):
        plt.plot(node_data[x_axis], np.log10(node_data['local_gap']), label=f'node {i}')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    fig.tight_layout(pad=2.)
    return fig

def plot_duality_gap(k, data, x_axis='i_iter', x_label='global iteration step', comp_data=None):
    # Plot primal and dual
    fig = plt.figure(f'Global Results, {k} nodes')
    gs = GridSpec(3, 2, figure=fig)

    fig.add_subplot(gs[0,0])
    plt.title('Primal')
    plt.xlabel(x_label)
    plt.ylabel(r'$\mathcal{P}(w)$')
    plt.plot(data[x_axis], data['P'], label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], comp_data['P'], label='CoCoA')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    fig.add_subplot(gs[0,1])
    plt.title('Dual')
    plt.xlabel(x_label)
    plt.ylabel(r'$\mathcal{D}(x)$')
    plt.plot(data[x_axis], data['D'], label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], comp_data['D'], label='CoCoA')

    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    fig.add_subplot(gs[1:,0:])
    plt.title('Primal-Dual Gap')
    plt.xlabel(x_label)
    plt.ylabel(r'$\log_{10}(global\ gap)$')
    plt.plot(data[x_axis], np.log10(np.abs(data['gap'])), label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], np.log10(np.abs(comp_data['gap'])), label='CoCoA')

    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    fig.tight_layout(pad=2.)
    return fig

def plot_primal_dual(k, data, x_axis='i_iter', x_label='global iteration step', comp_data=None):
    fig = plt.figure(f'Primal and Dual Values, {k} nodes')
    plt.subplot(121)
    plt.title('Primal')
    plt.xlabel(x_label)
    plt.ylabel(r'$\mathcal{P}(w)$')
    plt.plot(data[x_axis], data['P'])
    
    plt.subplot(122)
    plt.title('Dual')
    plt.xlabel(x_label)
    plt.ylabel(r'$\mathcal{D}(x)$')
    plt.plot(data[x_axis], data['D'])

    fig.tight_layout(pad=2.)
    return fig

def plot_minimizers(k, data, x_axis='i_iter', x_label='global iteration step', comp_data=None):
    fig = plt.figure(f'Minimizer Values, {k} nodes')
    plt.title('Minimizer functions per iteration')
    plt.subplot(221)
    plt.xlabel(x_label)
    plt.ylabel(r'$f(Ax)$')
    plt.plot(data[x_axis], data['f'], label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], comp_data['f'], label='CoCoA')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.subplot(222)
    plt.xlabel(x_label)
    plt.ylabel(r'$g(x)$')
    plt.plot(data[x_axis], data['g'], label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], comp_data['g'], label='CoCoA')

    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.subplot(223)
    plt.xlabel(x_label)
    plt.ylabel(r'$f^*(w)$')
    plt.plot(data[x_axis], data['f_conj'], label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], comp_data['f_conj'], label='CoCoA')
    
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.subplot(224)
    plt.xlabel(x_label)
    plt.ylabel(r'$g^*(-w^TA)$')
    plt.plot(data[x_axis], data['g_conj'], label='CoLA')
    if comp_data is not None:
        plt.plot(comp_data[x_axis], comp_data['g_conj'], label='CoCoA')

    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    fig.tight_layout(pad=2.)
    return fig

def plot_update_and_global(local_, global_, global_y='gap', global_y_label='Global Gap', x_axis='i_iter', x_label='global iteration step'):
    fig = plt.figure()

    plt.title(r'$\|\|\Delta x_k\|\|$ and ' + global_y_label + f', {len(local_)} nodes')
    plt.xlabel(x_label)
    for (i, node_data) in enumerate(local_):
        plt.plot(node_data[x_axis], np.log10(node_data['delta_xk']), label=r'$\|\| \Delta x_k \|\|$'+f', node {i}')

    plt.plot(global_[x_axis], np.log10(np.abs(global_[global_y])), label=global_y_label)
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    fig.tight_layout(pad=2.)
    return fig


@click.command()
@click.option('--k', type=click.INT, default=None, help='Number of mpi processes')
@click.option('--logdir', type=click.STRING, default='log', help='root directory of output files')
@click.option('--dataset', type=click.STRING, default='rderms', help='dataset name')
@click.option('--compare', is_flag=True)
@click.option('--save', is_flag=True)
@click.option('--show/--no-show', default=True)
@click.option('--savedir', type=click.STRING, default=None)
def plot_results(k, logdir, dataset, compare, save, show, savedir):
    if save and savedir is None:
        savedir = 'out'
    log_path = os.path.join(logdir, dataset) 
    comp_path = os.path.join(logdir, 'cocoa', dataset)
    if not os.path.exists(log_path):
        print(f"Log directory not found at '{log_path}'")
        exit(1)
    if compare and not os.path.exists(comp_path):
        print(f"Log directory not found at '{comp_path}'")
        exit(1)
    local_results = []
    if not k:
        k = 1
        while True:
            next_dir = os.path.join(log_path, f'{k}')
            if os.path.exists(next_dir):
                break
            k+=1
            if k>256:
                print('No logs found in directory.')
                exit(1)
    
    log_path = os.path.join(log_path, f'{k}')
    comp_path = os.path.join(comp_path, f'{k}')
    assert k is not None and k>0, 'logs not found, try specifying `logdir` and `k`' 

    weights_path = os.path.join(logdir, dataset, 'final_weight.npy')
    showres = os.path.exists(weights_path)
    
    res = None
    comp_res = None
    local_results = [pd.read_csv(os.path.join(log_path,f'{i}result.csv')).loc[1:] for i in range(k)]
    global_results = pd.read_csv(os.path.join(log_path, 'result.csv')).loc[1:]

    if showres:
        ref_weights = np.load(weights_path, allow_pickle=True)
        ref_norm = np.linalg.norm(ref_weights)
        niter = int(max(global_results['i_iter']))
        res = [(i, np.linalg.norm(np.load(wf, allow_pickle=True) - ref_weights)/ref_norm) 
            for (i, wf) in [(i, os.path.join(log_path, f'weight_epoch_{i}.npy')) for i in range(1,niter+1)] 
            if os.path.exists(wf)]
        if compare:
            comp_res = [(i, np.linalg.norm(np.load(wf, allow_pickle=True) - ref_weights)/ref_norm) \
                for (i, wf) in [(i, os.path.join(comp_path, f'weight_epoch_{i}.npy')) for i in range(1,niter+1)]
                if os.path.exists(wf)]
        res = pd.DataFrame([*res], columns=['i_iter', 'res'])
        print(res)
        if comp_res is not None: 
            comp_res = pd.DataFrame([*comp_res], columns=['i_iter', 'res'])

    comp_local = None 
    comp_global = None 
    
    if compare:
        comp_local = [pd.read_csv(os.path.join(comp_path,f'{i}result.csv')).loc[1:] for i in range(k)]
        comp_global = pd.read_csv(os.path.join(comp_path, 'result.csv')).loc[1:]
    else:
        comp_path = None
    
    if savedir is not None:
        savedir = os.path.join(savedir, dataset, f'{k}')
        os.makedirs(savedir, exist_ok=True)
    def saveorshow(fig, name):
        if save:
            fig.savefig(os.path.join(savedir, name), dpi=150)
        if show:
            plt.show()

    # Plotting
    fig = plot_local_results(k, local_results, x_label='Iteration Count', comp_data=comp_local)
    saveorshow(fig, 'local.png')

    fig = plot_minimizers(k, global_results, x_label='Iteration Count', comp_data=comp_global)
    saveorshow(fig, 'minimizers.png')

    fig = plot_duality_gap(k, global_results, x_label='Iteration Count', comp_data=comp_global)
    saveorshow(fig, 'duality-gap.png')

    fig = plot_update_and_global(local_results, global_results)
    saveorshow(fig, 'update-and-gap.png')

    if showres:
        fig = plot_update_and_global(local_results, res, global_y='res', global_y_label=r'$\log_{10} (\|\|x_k - x^*\|\|/\|\|x^*\|\|)$')
        saveorshow(fig, 'update-and-res.png')

        fig = plot_residual(k, res, comp_res)
        saveorshow(fig, 'relative_error.png') 

if __name__ == '__main__':
    plot_results()
