import click
import numpy as np
import pandas as pd
import cola.communication as comm

from cola.dataset import load_dataset, load_dataset_by_rank
from cola.graph import define_graph_topology
from cola.cocoasolvers import configure_solver
from cola.algo import Cola
from cola.monitor import Monitor
import pickle
import os, sys
sys.path.append('./scripts')
from create_report_plots import make_intercept_plots, clean_plots

def getGraphs(K, verbose=1):
    """ Returns a dictionary of graphs with relevant topologies """
    graphs = {}
    for topo in ['complete', 'ring', 'grid']:
        try:
            graphs[topo] = define_graph_topology(K, topo, verbose=verbose)
        except:
            graphs[topo] = None
    return graphs

def getSolversByLambda(l1_ratio, n_lambdas=10, size=1, random_state=42):
    lambdas = np.logspace(-n_lambdas+1, 0, n_lambdas)
    solvers = {}
    for lam in lambdas:
        solvers[lam] = configure_solver(name='ElasticNet', l1_ratio=l1_ratio, lambda_=lam/size, random_state=random_state)
    return solvers

@click.command('report_experiments')
@click.argument('dataset', type=click.STRING)
def main(dataset):
    random_state = 42

    # Fix gamma = 1.0 according to:
    #   Adding vs. Averaging in Distributed Primal-Dual Optimization
    gamma = 1.0
    theta = 1e-3
    global_iters = 500
    local_iters = 20
    # Initialize process group
    comm.init_process_group('mpi')

    # Get rank of current process
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    # Create graph with specified topology
    graphs_center = getGraphs(world_size-1)
    graphs_affine = getGraphs(world_size)


    dataset_path = os.path.join('data', dataset, 'features', f'{world_size-1}')
    if rank != world_size-1:
        comm.init_process_group('mpi', world_size-1)
        X, y, X_test, y_test = load_dataset_by_rank(dataset, rank, world_size-1, random_state=random_state, verbose=1)
    comm.init_process_group('mpi')
    Xones, yones, Xones_test, yones_test = load_dataset_by_rank(dataset+'-ones', rank, world_size, random_state=random_state, verbose=1)
    index = np.asarray(np.load(os.path.join(dataset_path, 'index.npy'), allow_pickle=True), dtype=np.int)
    index_test = np.asarray(np.load(os.path.join(dataset_path, 'index_test.npy'), allow_pickle=True), dtype=np.int)

    # Define subproblem
    # lasso_solvers = getSolversByLambda(1, n_lambdas=10, size=len(y), random_state=random_state)
    # elasticnet_solvers = getSolversByLambda(0.5, n_lambdas=10, size=len(y), random_state=random_state)
    # l2_solvers = getSolversByLambda(0, n_lambdas=10, size=len(y), random_state=random_state)
    solver = configure_solver(name='ElasticNet', l1_ratio=0.8, lambda_=3./len(yones), random_state=random_state)

    # Add hooks to log and save metrics.
    output_dir = os.path.join('out', 'report', dataset)
    clean_plots()
    # Run CoLA
    for topo in graphs_center:
        comm.barrier()
        if rank != world_size-1:
            if not graphs_center[topo]:
                continue
            comm.init_process_group('mpi', world_size-1)    
            suf = f'{world_size-1}-{topo}'

            mon_default = Monitor(output_dir, mode='all', verbose=1, Ak=X, Ak_test=X_test, y_test=y_test, name='Default')
            model_default = Cola(gamma, solver, theta, fit_intercept=False, normalize=True)
            mon_default.init(model_default, graphs_center[topo])
            model_default = model_default.fit(X, y, graphs_center[topo], mon_default, global_iters, local_iters)

            # Show test stats
            if rank == 0:
                print(f'Default - {topo}')
            mon_default.show_test_statistics()
            # Save final model
            mon_default.save(modelname=f'model-default-{suf}.pickle', logname=f'result-default-{suf}.csv')

            mon_center = Monitor(output_dir, mode='all', verbose=1, Ak=X, Ak_test=X_test, y_test=y_test, name='Center')
            model_center = Cola(gamma, solver, theta, fit_intercept=True, normalize=True)
            mon_center.init(model_center, graphs_center[topo])
            model_center = model_center.fit(X, y, graphs_center[topo], mon_center, global_iters, local_iters)

            # Show test stats
            if rank == 0:
                print(f'Center - {topo}')
            mon_center.show_test_statistics()
            
            # Save final model
            mon_center.save(modelname=f'model-center-{suf}.pickle', logname=f'result-center-{suf}.csv')
            comm.reset()
        else:
            mon_center = model_center = mon_default = model_default = None

        # Run CoLA
        if topo != 'grid':
            comm.barrier()
            suf = f'{world_size}-{topo}'
            mon_affine = Monitor(output_dir, mode='all', verbose=1, Ak=Xones, Ak_test=Xones_test, y_test=yones_test, name='Affine')
            model_affine = Cola(gamma, solver, theta, fit_intercept=False, normalize=True)
            mon_affine.init(model_affine, graphs_affine[topo])
            model_affine = model_affine.fit(Xones, yones, graphs_affine[topo], mon_affine, global_iters, local_iters)

            # Show test stats
            if rank == 0:
                print(f'Affine - {topo}')
            mon_affine.show_test_statistics()
            
            # Save final model
            mon_affine.save(modelname=f'model-affine-{suf}.pickle', logname=f'result-affine-{suf}.csv')
        else:
            mon_affine = None
        make_intercept_plots(f'intercept_{topo}_', mon_default, mon_center, mon_affine, index, index_test)

if __name__ == "__main__":
    main()