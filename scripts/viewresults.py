import click
import csv
from math import ceil
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import os

class PlotSpec:
    def __init__(self, title, yaxis, ylabel, log_y=True, xaxis='i_iter', xlabel='Iteration Count'):
        self.title = title
        self.xaxis = xaxis
        self.xlabel = xlabel
        self.yaxis = yaxis
        self.ylabel = ylabel
        self.log_y = log_y

    def _plot(self, ax, data, label='CoLA'):
        if self.log_y:
            ax.plot(data[self.xaxis], np.log10(np.abs(data[self.yaxis])), label=label)
        else:
            ax.plot(data[self.xaxis], data[self.yaxis], label=label)

    def plot(self, ax, data, islocal=False, label=''):
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if islocal:
            if label:
                label += ', '
            for (i, node) in enumerate(data):
                self._plot(ax, node, label=label+f'node {i}')
            
            ncol = int(np.ceil(len(data)/8))
            self.add_legend(ax, ncol)
        else:
            self._plot(ax, data, label=label)

    @classmethod
    def add_legend(cls, ax, ncol=1):
        leg = ax.legend(loc='best', ncol=ncol, shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)

class FigSpec:
    def __init__(self, name, data=None, comp_data=None, width=mpl.rcParams['figure.figsize'][0]):
        self.plots = []
        self.name = name
        self.fig = plt.figure(self.name)
        self.width = width
        self.data = data
        self.comp_data = comp_data

    def add_plot(self, spec, data=None, label=None, comp_spec=None, comp_data=None, comp_label=None, pos=None, sidebyside=False, islocal=False):
        if data is None:
            data = self.data
        if comp_data is None:
            comp_data = self.comp_data
        
        if label==None:
            label = label or '' if islocal else 'CoLA'
        plots = [{ 'spec': spec, 'label': label, 'data': data }]
        if comp_data is not None:
            if islocal:
                sidebyside = True
            if comp_spec is None:
                comp_spec = spec
            if comp_label==None:
                comp_label = label or '' if islocal else 'CoCoA'
            plots.append({
                'spec': comp_spec,
                'label': comp_label,
                'data': comp_data
            })
        
        self.plots.append({'local': islocal, 'sidebyside': sidebyside, 'pos': pos, 'data': plots})

    def draw(self):
        self.fig.clf()

        self.fig.suptitle(self.name, fontsize=16) 

        n_plots = len(self.plots)
        self.fig.set_size_inches(self.width, n_plots*self.width*.5)
        gs = self.fig.add_gridspec(len(self.plots), 2)
        for (i, plot) in enumerate(self.plots):
            if plot['sidebyside']:
                for (k, p) in enumerate(plot['data']):
                    pos = plot['pos'] or np.s_[i, k]
                    ax = self.fig.add_subplot(gs[pos])
                    p['spec'].plot(ax, p['data'], islocal=plot['local'], label=p['label'])
                    if not plot['local']:
                        p['spec'].add_legend(ax)
            else:
                pos = plot['pos'] or np.s_[i,:]
                ax = self.fig.add_subplot(gs[pos])
                for (k, p) in enumerate(plot['data']):
                    p['spec'].plot(ax, p['data'], islocal=plot['local'], label=p['label'])
                if not plot['local']:
                    PlotSpec.add_legend(ax)

        self.fig.tight_layout()
        
def create_PlotSpec(yaxis, xaxis='i_iter', xlabel='Iteration Count'):
    if yaxis=='res':
        return PlotSpec('Residuals', 'res', r'$\log_{10} (\|\|x_k - x^*\|\|/\|\|x^*\|\|)$', xaxis=xaxis, xlabel=xlabel)
    if yaxis=='local_gap':
        return PlotSpec('Gap', 'local_gap', r'$\log_{10} (local\ gap)$', xaxis=xaxis, xlabel=xlabel)
    if yaxis=='cv2':
        return PlotSpec('Consensus Violation', 'cv2', r'$\log_{10}(\|\| A_kx_k - v_k \|\|^2/ \|\|v_k\|\|^2)$', xaxis=xaxis, xlabel=xlabel)
    if yaxis=='delta_xk':
        return PlotSpec('Change in Local Iterates', 'delta_xk', r'$\log_{10}(\|\| \Delta x_k \|\|)$', xaxis=xaxis, xlabel=xlabel)
    if yaxis=='P':
        return PlotSpec('Primal', 'P', r'$\log_{10} \|\mathcal{P}(w)\|$', xaxis=xaxis, xlabel=xlabel)
    if yaxis=='D':
        return PlotSpec('Dual', 'D', r'$\log_{10} \|\mathcal{D}(x)\|$', xaxis=xaxis, xlabel=xlabel)
    if yaxis=='gap':
        return PlotSpec('Gap', 'gap', r'$\log_{10}\|\mathcal{P}(w) + \mathcal{D}(x)\|$', xaxis=xaxis, xlabel=xlabel)
    if yaxis=='f':
        return PlotSpec('f', 'f', r'$\log_{10}\|f(Ax)\|$', xaxis=xaxis, xlabel=xlabel)
    if yaxis=='g':
        return PlotSpec('g', 'g', r'$\log_{10}\|g(x)\|$', xaxis=xaxis, xlabel=xlabel)
    if yaxis=='f_conj':
        return PlotSpec('f*', 'f_conj', r'$\log_{10}\|f^*(w)\|$', xaxis=xaxis, xlabel=xlabel)
    if yaxis=='g_conj':
        return PlotSpec('g*', 'g_conj', r'$\log_{10}\|g^*(-w^TA)\|$', xaxis=xaxis, xlabel=xlabel)
    if yaxis=='rmse':
        return PlotSpec('Root Mean Squared Error', yaxis, r'rmse', xaxis=xaxis, xlabel=xlabel, log_y=False)
    if yaxis=='r2':
        return PlotSpec('$R^2$', yaxis, r'$R^2$', xaxis=xaxis, xlabel=xlabel, log_y=False)
    if yaxis=='max_rel':
        return PlotSpec('Maximum Relative Error', yaxis, r'max relative error', xaxis=xaxis, xlabel=xlabel, log_y=False)
    if yaxis=='l2_rel':
        return PlotSpec('Relative Error, 2-Norm', yaxis, r'$\|\|y^{pred}-y^{test}\|\|_2 / \|\|y^{test}\|\|_2$', xaxis=xaxis, xlabel=xlabel, log_y=False)

def plot_residual(k, res, comp_res=None):
    figspec = FigSpec(f'Residuals, {k} nodes')
    pltspec = create_PlotSpec('res')
    figspec.add_plot(res, pltspec, comp_data=comp_res)
    figspec.draw()
    return figspec.fig

def plot_local_results(k, local, xaxis='i_iter', xlabel='global iteration step', comp_data=None):
    figspec = FigSpec(f'Local Results, {k} nodes', data=local, comp_data=comp_data)

    gap = create_PlotSpec('local_gap', xaxis=xaxis, xlabel=xlabel)
    figspec.add_plot(gap, islocal=True)

    cv2 = create_PlotSpec('cv2', xaxis=xaxis, xlabel=xlabel)
    figspec.add_plot(cv2, islocal=True)

    delta_xk = create_PlotSpec('delta_xk', xaxis=xaxis, xlabel=xlabel)
    figspec.add_plot(delta_xk, islocal=True)
    
    figspec.draw()
    return figspec.fig

def plot_duality_gap(k, data, xaxis='i_iter', xlabel='global iteration step', comp_data=None):
    # Plot primal and dual
    fspec = FigSpec(f'Global Results, {k} nodes', data=data, comp_data=comp_data)

    primal = create_PlotSpec('P', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(primal)

    dual = create_PlotSpec('D', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(dual)

    gap = create_PlotSpec('gap', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(gap)

    fspec.draw()
    return fspec.fig

def plot_minimizers_end(k, data, xaxis='i_iter', xlabel='global iteration step', comp_data=None):
    n = data.shape[0]
    data_end = data.loc[n//2:]
    # comp_data_end = comp_data[n//2:, :] if comp_data is not None else None
    fspec = FigSpec(f'Minimizer Values, iters {n//2}-{n}, {k} nodes', data=data_end, comp_data=data_end)
    f = create_PlotSpec('f', xaxis=xaxis, xlabel=xlabel)
    g = create_PlotSpec('g', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(f, label='f', comp_spec=g, comp_label='g', sidebyside=True)

    f_conj = create_PlotSpec('f_conj', xaxis=xaxis, xlabel=xlabel)
    g_conj = create_PlotSpec('g_conj', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(f_conj, label='f*', comp_spec=g_conj, comp_label='g*', sidebyside=True)

    fspec.draw()
    return fspec.fig

def plot_minimizers(k, data, xaxis='i_iter', xlabel='global iteration step', comp_data=None):
    fspec = FigSpec(f'Minimizer Values, {k} nodes', data=data, comp_data=data)
    f = create_PlotSpec('f', xaxis=xaxis, xlabel=xlabel)
    g = create_PlotSpec('g', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(f, label='f', comp_spec=g, comp_label='g', sidebyside=True)

    f_conj = create_PlotSpec('f_conj', xaxis=xaxis, xlabel=xlabel)
    g_conj = create_PlotSpec('g_conj', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(f_conj, label='f*', comp_spec=g_conj, comp_label='g*', sidebyside=True)

    fspec.draw()
    return fspec.fig

def plot_minimizers_exact(k, data, xaxis='i_iter', xlabel='global iteration step', comp_data=None):
    data1 = data.copy()
    data1[:,'f'] = data[-1,'f'] - data[:,'f']
    data1[:,'g'] = data[-1,'g'] - data[:,'g']
    data1[:,'f_conj'] = data[-1,'f_conj'] - data[:,'f_conj']
    data1[:,'g_conj'] = data[-1,'g_conj'] - data[:,'g_conj']
    fspec = FigSpec(f'Minimizer Values, distance from minimum, {k} nodes', data=data1, comp_data=data1)
    f = PlotSpec('f', 'f', r'$\log_{10}\|f(Ax^*) - f(Ax)\|$', xaxis=xaxis, xlabel=xlabel)
    g = PlotSpec('g', 'g', r'$\log_{10}\|g(x^*) - g(x)\|$', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(f, label='f', comp_spec=g, comp_label='g', sidebyside=True)

    f_conj = create_PlotSpec('f_conj', xaxis=xaxis, xlabel=xlabel)
    g_conj = PlotSpec('g*', 'g_conj', r'$\log_{10}\|g^*(-w^TA)\|$', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(f_conj, label='f*', comp_spec=g_conj, comp_label='g*', sidebyside=True)

    fspec.draw()
    return fspec.fig

def plot_update_and_global(local_, global_, global_y='gap', global_ylabel='Global Gap', xaxis='i_iter', xlabel='global iteration step'):
    fspec = FigSpec(r'$\log_{10} \|\|\Delta x_k\|\|$ and ' + global_ylabel + f', {len(local_)} nodes')
    gap = PlotSpec('', 'gap', '', xaxis=xaxis, xlabel=xlabel)
    update = PlotSpec('', 'delta_xk', '', xaxis=xaxis, xlabel=xlabel)

    fspec.add_plot(gap, data=global_, label=r'$\log_{10} \|global\ gap\|$', pos=np.s_[:,:])
    fspec.add_plot(update, data=local_, label=r'$\log_{10} \|\|\Delta x_k\|\|$', islocal=True, pos=np.s_[:,:])
    
    fspec.draw()
    return fspec.fig

def plot_test_statistics(k, data, xaxis='i_iter', xlabel='global iteration step', comp_data=None):  
    n_train, n_test = np.max(data.loc[:,'n_train']), np.max(data.loc[:,'n_test'])
    fspec = FigSpec(f'Prediction Test Statistics ({n_train}/{n_test}/{n_train+n_test}), {k} nodes', data=data, comp_data=data)
    rmse = create_PlotSpec('rmse', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(rmse)
    r2 = create_PlotSpec('r2', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(r2)

    l2_rel = create_PlotSpec('l2_rel', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(l2_rel)
    max_rel = create_PlotSpec('max_rel', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(max_rel)

    fspec.draw()
    return fspec.fig

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
    fig = plot_local_results(k, local_results, comp_data=comp_local)
    saveorshow(fig, 'local.png')

    fig = plot_minimizers(k, global_results, comp_data=comp_global)
    saveorshow(fig, 'minimizers.png')

    fig = plot_minimizers_end(k, global_results, comp_data=comp_global)
    saveorshow(fig, 'minimizers-last-half.png')

    fig = plot_duality_gap(k, global_results, comp_data=comp_global)
    saveorshow(fig, 'duality-gap.png')

    fig = plot_update_and_global(local_results, global_results)
    saveorshow(fig, 'update-and-gap.png')

    if 'n_test' in global_results:
        fig = plot_test_statistics(k, global_results, comp_data=comp_global)
        saveorshow(fig, 'test_statistics.png')

    if showres:
        fig = plot_update_and_global(local_results, res, global_y='res', global_ylabel=r'$\log_{10} (\|\|x_k - x^*\|\|/\|\|x^*\|\|)$')
        saveorshow(fig, 'update-and-res.png')

        fig = plot_residual(k, res, comp_res)
        saveorshow(fig, 'relative_error.png') 

if __name__ == '__main__':
    plot_results()
