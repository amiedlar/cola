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

class PlotSpec:
    def __init__(self, title, yaxis, ylabel, log_y=True, xaxis='i_iter', xlabel='Iteration Count'):
        self.title = title
        self.xaxis = xaxis
        self.xlabel = xlabel
        self.yaxis = yaxis
        self.ylabel = ylabel
        self.log_y = log_y
        if self.log_y and self.ylabel:
            self.ylabel = r"$\log_{10}($" + self.ylabel + r"$)$"

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

    def draw(self, override_height=None, large=False, flip=False):
        self.fig.clf()

        if large:
            plt.rc('font', size=24)
            # plt.rc('axes', titlesize='large')
            # plt.rc('axes', labelsize='large')
            # plt.rc('xtick', labelsize='medium')    # fontsize of the tick labels
            # plt.rc('ytick', labelsize='medium')    # fontsize of the tick labels
            # plt.rc('legend', fontsize='large')    # legend fontsize
            # plt.rc('figure', titlesize='x-large')  # fontsize of the figure title
        self.fig.suptitle(self.name)#, fontsize='x-large') 

        n_plots = len(self.plots)
        mult = 2 if large else 1 
        if override_height is None:
            override_height=n_plots
        if flip:
            self.fig.set_size_inches(32,12)
            gs = self.fig.add_gridspec(1, override_height)
        else:
            self.fig.set_size_inches(mult*self.width, mult*override_height*self.width*.5)
            gs = self.fig.add_gridspec(override_height, 2)
        for (i, plot) in enumerate(self.plots):
            if plot['sidebyside']:
                for (k, p) in enumerate(plot['data']):
                    pos = plot['pos'] or (np.s_[k, i] if flip else np.s_[i, k])
                    ax = self.fig.add_subplot(gs[pos])
                    p['spec'].plot(ax, p['data'], islocal=plot['local'], label=p['label'])
                    if not plot['local']:
                        p['spec'].add_legend(ax)
            else:
                pos = plot['pos'] or (np.s_[:,i] if flip else np.s_[i,:])
                ax = self.fig.add_subplot(gs[pos])
                for (k, p) in enumerate(plot['data']):
                    p['spec'].plot(ax, p['data'], islocal=plot['local'], label=p['label'])
                if not plot['local']:
                    PlotSpec.add_legend(ax)

        self.fig.tight_layout()
        
def create_PlotSpec(yaxis, xaxis='i_iter', xlabel='Iteration Count', **kwargs):
    if yaxis=='res':
        return PlotSpec('Residuals', yaxis, r'$\|\|x_k - x^*\|\|/\|\|x^*\|\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='local_gap':
        return PlotSpec('Gap', yaxis, r'local gap', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='cert_gap':
        return PlotSpec('Gap', yaxis, r'$ v_k^\top\nabla f(v_k) + \sum_{i\in\mathcal{P}_k} (g_i(v_k) + g_i^* (-A_i^\top \nabla f(v_k)))$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='cv2':
        return PlotSpec('Consensus Violation', yaxis, r'$\|\| A_kx_k - v_k \|\|^2/ \|\|v_k\|\|^2$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='cert_cv':
        return PlotSpec('Consensus Violation', yaxis, r'$\|\| \nabla f(v_k) - \frac{1}{\|\mathcal{N}_k \|} \sum_{j\in\mathcal{N}_k}\nabla f(v_j) \|\|_2$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='delta_xk':
        return PlotSpec('Change in Local Iterates', yaxis, r'$\|\| \Delta x_k \|\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='P':
        return PlotSpec('Primal', yaxis, r'$\|\mathcal{P}(x)\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='D':
        return PlotSpec('Dual', yaxis, r'$\|\mathcal{D}(w)\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='gap':
        return PlotSpec('Gap', yaxis, r'$\|\mathcal{P}(x) + \mathcal{D}(w)\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='gap_rel':
        return PlotSpec('Relative Gap', yaxis, r'$\|\mathcal{P}(x) + \mathcal{D}(w)\|/\|\mathcal{D}(w)\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='f':
        return PlotSpec('f', yaxis, r'$\|f(Ax)\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='g':
        return PlotSpec('g', yaxis, r'$\|g(x)\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='f_conj':
        return PlotSpec('f*', yaxis, r'$\|f^*(w)\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='g_conj':
        return PlotSpec('g*', yaxis, r'$\|g^*(-w^TA)\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='rmse':
        return PlotSpec('RMSE', yaxis, r'rmse', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='r2':
        return PlotSpec('R^2', yaxis, r'R^2', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='max_rel':
        return PlotSpec('max. rel. error', yaxis, r'max. error (V)', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='l2_rel':
        return PlotSpec('2-Norm Relative Error', yaxis, r'$\|\|y^{pred}-y^{test}\|\|_2 / \|\|y^{test}\|\|_2$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='f_min_final':
        return PlotSpec('f', yaxis, r'$\|f(Ax^*) - f(Ax)\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='g_min_final':
        return PlotSpec('g', yaxis, r'$\|g(x^*) - g(x)\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='f_conj_min_final':
        return PlotSpec('f*', yaxis, r'$\|f^*(w^*) - f^*(w(x))\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)
    if yaxis=='g_conj_min_final':
        return PlotSpec('g*', yaxis, r'$\|g^*(-(w^*)^TA)-g^*(-w(x)^TA)\|$', xaxis=xaxis, xlabel=xlabel, **kwargs)

def plot_residual(k, res, comp_res=None, large=False):
    figspec = FigSpec(f'Residuals, {k} nodes')
    pltspec = create_PlotSpec('res')
    figspec.add_plot(res, pltspec, comp_data=comp_res)
    figspec.draw(large=large)
    return figspec.fig

def plot_local_results(k, local, xaxis='i_iter', xlabel='global iteration step', comp_data=None, large=False):
    figspec = FigSpec(f'Local Results, {k} nodes', data=local, comp_data=comp_data)

    gap = create_PlotSpec('local_gap', xaxis=xaxis, xlabel=xlabel)
    figspec.add_plot(gap, islocal=True)

    cv2 = create_PlotSpec('cv2', xaxis=xaxis, xlabel=xlabel)
    figspec.add_plot(cv2, islocal=True)

    delta_xk = create_PlotSpec('delta_xk', xaxis=xaxis, xlabel=xlabel)
    figspec.add_plot(delta_xk, islocal=True)
    
    figspec.draw(large=large)
    return figspec.fig

def plot_local_cert(k, local, xaxis='i_iter', xlabel='global iteration step', comp_data=None, large=False):
    figspec = FigSpec(f'Local Certificates, {k} nodes', data=local, comp_data=comp_data)

    gap = create_PlotSpec('cert_gap', xaxis=xaxis, xlabel=xlabel, log_y=True)
    figspec.add_plot(gap, islocal=True)

    cv2 = create_PlotSpec('cert_cv', xaxis=xaxis, xlabel=xlabel, log_y=True)
    figspec.add_plot(cv2, islocal=True)
    
    figspec.draw(large=large)
    return figspec.fig

def plot_duality_gap(k, data, xaxis='i_iter', xlabel='global iteration step', comp_data=None, large=False):
    # Plot primal and dual
    fspec = FigSpec(f'Global Results, {k} nodes', data=data, comp_data=comp_data)

    primal = create_PlotSpec('P', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(primal)

    dual = create_PlotSpec('D', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(dual)

    gap = create_PlotSpec('gap', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(gap)

    gap_rel = create_PlotSpec('gap_rel', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(gap_rel)

    fspec.draw(large=large)
    return fspec.fig

def plot_minimizers_end(k, data, xaxis='i_iter', xlabel='global iteration step', comp_data=None, large=False):
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

    fspec.draw(large=large)
    return fspec.fig

def plot_minimizers(k, data, xaxis='i_iter', xlabel='global iteration step', comp_data=None, large=False):
    fspec = FigSpec(f'Minimizer Values, {k} nodes', data=data, comp_data=data)
    f = create_PlotSpec('f', xaxis=xaxis, xlabel=xlabel)
    g = create_PlotSpec('g', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(f, label='f', comp_spec=g, comp_label='g', sidebyside=True)

    f_conj = create_PlotSpec('f_conj', xaxis=xaxis, xlabel=xlabel)
    g_conj = create_PlotSpec('g_conj', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(f_conj, label='f*', comp_spec=g_conj, comp_label='g*', sidebyside=True)

    fspec.draw(large=large)
    return fspec.fig

def plot_minimizers_exact(k, data, xaxis='i_iter', xlabel='global iteration step', comp_data=None, large=False):
    fspec = FigSpec(f'Minimizer Values, distance from minimum, {k} nodes', data=data, comp_data=data)

    f = create_PlotSpec('f_min_final', xaxis=xaxis, xlabel=xlabel) 
    g = create_PlotSpec('g_min_final', xaxis=xaxis, xlabel=xlabel) 
    fspec.add_plot(f, label='f', comp_spec=g, comp_label='g', sidebyside=True)

    f_conj = create_PlotSpec('f_conj_min_final', xaxis=xaxis, xlabel=xlabel)
    g_conj = create_PlotSpec('g_conj_min_final', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(f_conj, label='f*', comp_spec=g_conj, comp_label='g*', sidebyside=True)

    fspec.draw(large=large)
    return fspec.fig

def plot_update_and_global(local_, global_, global_y='gap', global_ylabel='Global Gap', xaxis='i_iter', xlabel='global iteration step', large=False):
    fspec = FigSpec(r'$\log_{10} \|\|\Delta x_k\|\|$ and ' + global_ylabel + f', {len(local_)} nodes')
    gap = PlotSpec('', 'gap', '', xaxis=xaxis, xlabel=xlabel)
    update = PlotSpec('', 'delta_xk', '', xaxis=xaxis, xlabel=xlabel)

    fspec.add_plot(gap, data=global_, label=r'$\log_{10} \|global\ gap\|$', pos=np.s_[:,:])
    fspec.add_plot(update, data=local_, label=r'$\log_{10} \|\|\Delta x_k\|\|$', islocal=True, pos=np.s_[:,:])
    
    fspec.draw(large=large)
    return fspec.fig

def plot_cert_and_global(local_, global_, global_y='gap', global_ylabel='Global Gap', xaxis='i_iter', xlabel='global iteration step', large=False):
    fspec = FigSpec(f'Local Certificates and Global Gap , {len(local_)} nodes')
    gap = PlotSpec('', 'gap', '', xaxis=xaxis, xlabel=xlabel)
    cert_gap = PlotSpec('Local and Global Gaps', 'cert_gap_scaled', '', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(gap, data=global_, label=r'$\log_{10} \|global\ gap\|$', pos=np.s_[0:2,:])
    fspec.add_plot(cert_gap, data=local_, label=r'$\log_{10} \|cert\ gap\|$', islocal=True, pos=np.s_[0:2,:])
    
    cert_cv = PlotSpec('Local CV and Global Gap', 'cert_cv_scaled', '', xaxis=xaxis, xlabel=xlabel)
    fspec.add_plot(gap, data=global_, label=r'$\log_{10} \|global\ gap\|$', pos=np.s_[2:4,:])
    fspec.add_plot(cert_cv, data=local_, label=r'$\log_{10} \|cert\ cv\|$', islocal=True, pos=np.s_[2:4,:])
    fspec.draw(large=large)
    return fspec.fig

def plot_test_statistics(k, data, xaxis='i_iter', xlabel='global iteration step', comp_data=None, large=False):  
    n_train, n_test = np.max(data.loc[:,'n_train']), np.max(data.loc[:,'n_test'])
    fspec = FigSpec(f'Prediction Test Statistics ({n_train}/{n_test}/{n_train+n_test}), {k} nodes', data=data, comp_data=None)
    rmse = create_PlotSpec('rmse', xaxis=xaxis, xlabel=xlabel, log_y=False)
    fspec.add_plot(rmse)
    r2 = create_PlotSpec('r2', xaxis=xaxis, xlabel=xlabel, log_y=False)
    fspec.add_plot(r2)

    l2_rel = create_PlotSpec('l2_rel', xaxis=xaxis, xlabel=xlabel, log_y=False)
    fspec.add_plot(l2_rel)
    max_rel = create_PlotSpec('max_rel', xaxis=xaxis, xlabel=xlabel, log_y=False)
    fspec.add_plot(max_rel)

    fspec.draw(large=large)
    return fspec.fig

def plot_linear_regression(k, dataset, logdir, check_iter=None, large=False):
    data_dir = os.path.join('data', dataset, 'features', str(k))
    index = np.asarray(np.load(os.path.join(data_dir, 'index.npy'), allow_pickle=True), dtype=np.int)
    index_test = np.asarray(np.load(os.path.join(data_dir, 'index_test.npy'), allow_pickle=True), dtype=np.int)
    col_perm = np.asarray(np.load(os.path.join(data_dir, 'col_index.npy'), allow_pickle=True), dtype=np.int)
    from sklearn.datasets import load_svmlight_file
    X, y = load_svmlight_file(os.path.join('..', 'data', dataset+'.svm'))
    if check_iter is None:
        weights = np.load(os.path.join(logdir, f'weight.npy'), allow_pickle=True)
    else:
        weights = np.load(os.path.join(logdir, f'weight_epoch_{check_iter}.npy'), allow_pickle=True)
    print(weights)
    regression = X[:,col_perm]*weights
    t = np.linspace(0, len(y)//4, len(y))

    c_train = (0.95294118, 0.56078431, 0.14901961)
    c_test = (0.39215686, 0.77254902, 0.93333333)
    c_reg = (0.29411765, 0.32941176, 0.36470588)
    if large:
        plt.rc('font', size=24, weight=300)
        # plt.rc('xlabel', )
        plt.rc('xtick', labelsize='large')
        # plt.rc('ylabel', )
        plt.rc('ytick', labelsize='large')
    fig = plt.figure('Linear regression')
    if large:
        fig.set_size_inches(16,10)
        
    
    # plt.title(f'Linear Regression after 1 Iteration for 5 {os.path.basename(logdir).capitalize()} Connected Inverters')
    plt.xlabel('Time (h)', fontsize='xx-large')#, fontsize=24)
    plt.ylabel('Voltage (V)', fontsize='xx-large')#, fontsize=24)
    # plt.tick_params(which='major', labelsize='large')
    train = plt.scatter(t[index], y[index], c=c_train, s=49, label=f'Train PCC Voltage ({len(index)})')
    test = plt.scatter(t[index_test], y[index_test], c=c_test, s=49, label=f'Test PCC Voltage ({len(index_test)})')
    plt.plot(t, regression, color=c_reg, label=f"Predicted PCC Voltage", linewidth=4)

    plt.legend(loc='best', fontsize='x-large')
    
    fig.tight_layout()
    return fig

@click.group('view-results')
def view_results():
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

@view_results.command('plot-results')
@click.option('--k', type=click.INT, default=None, help='Number of mpi processes')
@click.option('--logdir', type=click.STRING, default='log', help='root directory of output files')
@click.option('--dataset', type=click.STRING, default='rderms', help='dataset name')
@click.option('--topology', type=click.STRING, default='complete', help='graph topology')
@click.option('--compare', is_flag=True)
@click.option('--compdir', type=click.STRING, default='log/cocoa')
@click.option('--save', is_flag=True)
@click.option('--savedir', type=click.STRING, default=None)
@click.option('--show/--no-show', default=True)
@click.option('--large/--no-large')
@click.option('--linreg-iter', default=None, type=click.INT)
def plot_results(k, logdir, dataset, topology, compare, compdir, save, savedir, show, large, linreg_iter):
    if save and savedir is None:
        savedir = 'out'
    log_path = os.path.join(logdir, dataset) 
    comp_path = os.path.join(compdir, dataset)
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
            next_dir = os.path.join(log_path, f'{k:0>2}')
            if os.path.exists(next_dir):
                break
            k+=1
            if k>256:
                print('No logs found in directory.')
                exit(1)
    
    log_path = os.path.join(log_path, f'{k:0>2}', topology)
    comp_path = os.path.join(comp_path, f'{k:0>2}', topology)
    assert k is not None and k>0, 'logs not found, try specifying `logdir` and `k`' 

    weights_path = os.path.join(logdir, dataset, 'final_weight.npy')
    showres = os.path.exists(weights_path)
    
    res = None
    comp_res = None
    local_results = [pd.read_csv(os.path.join(log_path,f'{i}result.csv')).loc[1:] for i in range(k)]
    global_results = pd.read_csv(os.path.join(log_path, 'result.csv')).loc[1:]
    last = np.max(global_results['i_iter'])
    global_results['f_min_final'] = global_results['f'][last] - global_results['f']
    global_results['g_min_final'] = global_results['g'][last] - global_results['g']
    global_results['f_conj_min_final'] = global_results['f_conj'][last] - global_results['f_conj']
    global_results['g_conj_min_final'] = global_results['g_conj'][last] - global_results['g_conj']

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
        savedir = os.path.join(savedir, dataset, f'{k:0>2}', topology)
        os.makedirs(savedir, exist_ok=True)
    def saveorshow(fig, name):
        if save:
            fig.savefig(os.path.join(savedir, name), dpi=150, transparent=large)
        if show:
            plt.show()

    # Plotting
    fig = plot_local_results(k, local_results, comp_data=comp_local, large=large)
    saveorshow(fig, 'local.png')

    fig = plot_local_cert(k, local_results, comp_data=comp_local, large=large)
    saveorshow(fig, 'local-cert.png')

    fig = plot_minimizers(k, global_results, comp_data=comp_global, large=large)
    saveorshow(fig, 'minimizers.png')

    fig = plot_minimizers_end(k, global_results, comp_data=comp_global, large=large)
    saveorshow(fig, 'minimizers-last-half.png')

    fig = plot_minimizers_exact(k, global_results, comp_data=comp_global, large=large)
    saveorshow(fig, 'minimizers-exact.png')

    fig = plot_duality_gap(k, global_results, comp_data=comp_global, large=large)
    saveorshow(fig, 'duality-gap.png')

    fig = plot_update_and_global(local_results, global_results, large=large)
    saveorshow(fig, 'update-and-gap.png')

    fig = plot_cert_and_global(local_results, global_results, large=large)
    saveorshow(fig, 'cert-and-gap.png')

    if 'n_test' in global_results:
        fig = plot_test_statistics(k, global_results, comp_data=comp_global, large=large)
        saveorshow(fig, 'test_statistics.png')

        fig = plot_linear_regression(k, dataset, log_path, large=large, check_iter=linreg_iter)
        saveorshow(fig, 'linear_regression.png')

    if showres:
        fig = plot_update_and_global(local_results, res, global_y='res', global_ylabel=r'$\log_{10} (\|\|x_k - x^*\|\|/\|\|x^*\|\|)$')
        saveorshow(fig, 'update-and-res.png')

        fig = plot_residual(k, res, comp_res)
        saveorshow(fig, 'relative_error.png') 

@view_results.command('topology')
@click.option('--k', type=click.INT, default=None, help='Number of mpi processes')
@click.option('--dataset', type=click.STRING, default='rderms', help='dataset name')
@click.option('--logdir', type=click.STRING, default='log', help='root directory of output files')
@click.option('--save', is_flag=True)
@click.option('--savedir', type=click.STRING, default=None)
@click.option('--show/--no-show', default=True)
@click.option('--miniter', type=click.INT, default=1)
@click.option('--maxiter', type=click.INT, default=None)
@click.option('--large/--no-large', default=False)
def topology(k, dataset, logdir, save, savedir, show, miniter, maxiter, large):
    if save and savedir is None:
        savedir = 'out'
    save_path = os.path.join(savedir, dataset)
    os.makedirs(save_path, exist_ok=True)
    def saveorshow(fig, name):
        if save:
            fig.savefig(os.path.join(save_path, name), dpi=150)
            print(f"Plots saved to {os.path.join(save_path, name)}")
        if show:
            plt.show()
    if k is None:
        save_path = os.path.join(savedir, dataset, 'topology')
        os.makedirs(save_path, exist_ok=True)
        _topology_all_rank(dataset, logdir, saveorshow, large=large)
    else:
        save_path = os.path.join(save_path, f'{k:0>2}', 'topology')
        os.makedirs(save_path, exist_ok=True)
        _topology_single_rank(k, dataset, logdir, saveorshow, iterslice=np.s_[miniter:maxiter,:], large=large)

    
        
def _topology_single_rank(k, dataset, logdir, saveorshow, iterslice=np.s_[:,:], large=False):
    log_path = os.path.join(logdir, dataset, f'{k:0>2}')

    topologies = [f.path for f in os.scandir(log_path) if f.is_dir()]
    if len(topologies) == 0:
        print('no logs found')
        return

    results = [(os.path.basename(path), pd.read_csv(os.path.join(path, 'result.csv')).loc[iterslice]) 
                for path in topologies]
    fspec = FigSpec('Topology - Test Statistics')
    print(iterslice)
    print(results[0][1].shape)
    n_train, n_test = np.max(results[0][1].loc[:,'n_train']), np.max(results[0][1].loc[:,'n_test'])
    fspec = FigSpec(f'Prediction Test Statistics ({n_train}/{n_test}/{n_train+n_test}), {k} nodes')
    rmse = create_PlotSpec('rmse', log_y=False)
    r2 = create_PlotSpec('r2', log_y=False)
    l2_rel = create_PlotSpec('l2_rel', log_y=False)
    max_rel = create_PlotSpec('max_rel', log_y=False)
    for (topo, data) in results:
        fspec.add_plot(rmse, data=data, label=topo, pos=np.s_[0,:])
        fspec.add_plot(r2, data=data, label=topo, pos=np.s_[1,:])
        #fspec.add_plot(l2_rel, data=data, label=topo, pos=np.s_[2,:])
        fspec.add_plot(max_rel, data=data, label=topo, pos=np.s_[2,:])
    fspec.draw(override_height=3, large=large)
    saveorshow(fspec.fig, f'topology_test-stats_{n_train}-{n_test}-{n_train+n_test}_{k:0>2}-nodes.png')

def _topology_all_rank(dataset, logdir, saveorshow, large=False):
    log_path = os.path.join(logdir, dataset)
    ranks = [f.path for f in os.scandir(log_path) if f.is_dir()]
    if len(ranks) == 0:
        print('No logs found')
        return
    topologies = [(os.path.basename(r_path), [f.path for f in os.scandir(r_path) if f.is_dir()]) for r_path in ranks]
    topo_names = list(set([os.path.basename(f) for r in topologies for f in r[1]]))
    topo_names.sort()
    topologies = dict(zip(topo_names, map(lambda name: list([(worldsize, f) for (worldsize,r) in topologies for f in r if os.path.basename(f)==name]), topo_names)))
    results = dict(zip(topo_names, [None]*len(topo_names)))
    keys = None
    for topo in topologies:
        t_results = None
        for (worldsize, f) in topologies[topo]:
            df = pd.read_csv(os.path.join(f, 'result.csv'))
            max_iter = np.max(df['i_iter'])
            df = df.loc[max_iter,:]
            print(f"iter: {df['i_iter']}")
            df = df.append(pd.Series([int(worldsize)], index=['worldsize']))
            if keys is None:
                keys = df.index
            print(f"added data for worldsize {int(worldsize)}, topology '{os.path.basename(f)}'")
            t_data = np.reshape(np.asarray(df), (1, len(keys)))
            if t_results is None:
                t_results = t_data
            else:
                t_results = np.concatenate([t_results, t_data])
        results[topo] = np.asarray(t_results)
    
    for (topo, data) in results.items():
        results[topo] = pd.DataFrame(data, columns=keys)
    
    fspec = FigSpec('Topology and Worldsize - Test Statistics')
    t_list = list(results.values())
    n_train, n_test = np.max(t_list[0].loc[:,'n_train']), np.max(t_list[0].loc[:,'n_test'])
        
    fspec = FigSpec(f'Prediction Test Statistics ({n_train}/{n_test}/{n_train+n_test}), all MPI world sizes')
    rmse = create_PlotSpec('rmse', log_y=False, xaxis='worldsize', xlabel='# MPI Nodes')
    r2 = create_PlotSpec('r2', log_y=False, xaxis='worldsize', xlabel='# MPI Nodes')
    l2_rel = create_PlotSpec('l2_rel', log_y=False, xaxis='worldsize', xlabel='# MPI Nodes')
    max_rel = create_PlotSpec('max_rel', log_y=False, xaxis='worldsize', xlabel='# MPI Nodes')
    for (topo, data) in results.items():
        print(f"{topo}: RMSE={data['rmse']}")
        fspec.add_plot(rmse, data=data, label=topo, pos=np.s_[0,:])
        fspec.add_plot(r2, data=data, label=topo, pos=np.s_[1,:])
        fspec.add_plot(l2_rel, data=data, label=topo, pos=np.s_[2,:])
        fspec.add_plot(max_rel, data=data, label=topo, pos=np.s_[3,:])
    
    fspec.draw(override_height=4, large=large)

    saveorshow(fspec.fig, f'topology_test-stats_{n_train}-{n_test}-{n_train+n_test}_all-nodes.png')




if __name__ == '__main__':
    view_results()
