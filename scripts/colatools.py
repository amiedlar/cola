from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import normalize
from scipy.sparse import csc_matrix
import numpy as np
import os
import click


class RankEditor(object):
    def __init__(self, indir=None, outdir=False):
        self.indir = os.path.abspath(indir or '../data')
        self.outdir = os.path.abspath(outdir or 'data')
        self.data = None
        self.y = None
        

pass_editor = click.make_pass_decorator(RankEditor)

@click.group(chain=True)
@click.option('--indir', default=os.path.join('..', 'data'))
@click.option('--outdir', default='data')
@click.pass_context
def cli(ctx, indir, outdir):
    ctx.obj = RankEditor(indir, outdir) 

@cli.command('load')
@click.argument('dataset')
@click.option('--confirm', is_flag=True)
@pass_editor
def load(editor, dataset, confirm):
    if editor.data is not None and not confirm:
        print("There is existing data, pass --confirm flag to load anyway")
        return False
    
    if '.svm' not in dataset:
        dataset += '.svm'
    path = os.path.join(editor.indir, dataset)
    assert os.path.exists(path), f"SVM file '{path}' not found"
    old_data = editor.data.copy() if editor.data is not None else None
    old_y = editor.y.copy() if editor.y is not None else None
    try:
        editor.data, editor.y = load_svmlight_file(path)
        editor.data = editor.data.tocsc()
        print(f"Loaded '{dataset}', shape {editor.data.shape}")
        return
    except Exception as e:
        print(e)
        editor.data = old_data
        editor.y = old_y
        return

@cli.command('replace-column')
@click.argument('col', type=click.INT)
@click.argument('scheme', type=click.Choice(['uniform', 'scale', 'weights']))
@click.option('--scale-col', default=0, help="scale specified column (default 0)")
@click.option('--scale-by', default=1.0, help="scale factor for vector specified by `--scale-col` (default 1)")
@click.option('--weights', type=click.STRING, default=None,
    help="string containing python array with length n_col."
         "values in array correspond weights of each remaining column for replacement linear combination.")
@pass_editor
def replace_column(editor, col, scheme, scale_col, scale_by, weights):
    assert editor.data is not None, "load data before attempting to edit"
    assert not(weights is None and scheme == 'weights'), "specify weighting scheme"
    
    n_row, n_col = editor.data.shape
    if scheme == 'weights':
        weights = np.fromstring(weights, sep=' ')
        debug_str=f'A*{np.array2string(weights)}^T'
    elif scheme == 'scale':
        weights = np.zeros((n_col-1,))
        weights[scale_col] = scale_by
        debug_str=f'{scale_by}*A[:,{scale_col}]'
    elif scheme == 'uniform':
        weights = np.array([1/(n_col-1)]*(n_col-1))
        debug_str=f'average of other columns'
    else:
        return NotImplementedError

    weights = np.insert(weights, col, 0)

    new_col = editor.data * weights
    from scipy.sparse import csc_matrix
    editor.data[:,col] = csc_matrix(new_col.reshape((n_row,1)))
    print(f"Replaced A[:,{col}] with "+debug_str)
    return

@cli.command('insert-columns')
@click.argument('n', type=click.INT)
@click.option('--weights', type=click.STRING, default=None)
@pass_editor
def insert_columns(editor, n, weights):
    assert editor.data is not None, "load data before attempting to edit"
    # assert weights is not None or uniform, "either specify weights or use the `--uniform` flag"
    if weights:
        from json import loads
        weights = loads(weights)
        for spec in weights:
            _insert_column(editor, spec.get('scheme'), spec.get('scale_col', 0), spec.get('scale_by', 1), spec.get('weights'))
        return
    
    for i in range(n):
        _insert_column(editor, 'uniform', 0, 1, None)    

    return

@cli.command('insert-column')
@click.argument('scheme', type=click.Choice(['uniform', 'scale', 'weights']))
@click.option('--scale-col', default=0, help="scale specified column (default 0)")
@click.option('--scale-by', default=1.0, help="scale factor for vector specified by `--scale-col` (default 1)")
@click.option('--weights', type=click.STRING, default=None,
    help="string containing python array with length n_col."
         "values in array correspond weights of each remaining column for replacement linear combination.")
@pass_editor
def insert_column(editor, scheme, scale_col, scale_by, weights):
    _insert_column(editor, scheme, scale_col, scale_by, weights)

def _insert_column(editor, scheme, scale, scale_by, weights):
    assert editor.data is not None, "load data before attempting to edit"
    assert not (weights is None and scheme == 'weights'), "specify weighting scheme"
    
    n_row, n_col = editor.data.shape
    debug_str=''
    if scheme == 'weights':
        weights = np.fromstring(weights, sep=' ')
        debug_str=f'A*{np.array2string(weights)}^T'
    elif scheme == 'scale':
        weights = np.zeros((n_col,))
        weights[scale] = scale_by
        debug_str=f'{scale_by}*A[:,{scale}]'
    elif scheme == 'uniform':
        weights = np.array([1/n_col]*n_col)
        debug_str=f'average of existing columns'
    else:
        return NotImplementedError

    if weights.shape[0] != n_col:
        weights = np.concatenate((weights, [0]*(n_col - weights.shape[0])))
    weights = weights.reshape((n_col,1))
    
    from scipy.sparse import csc_matrix
    new_col = editor.data * weights
    new_col = new_col.reshape((n_row,1))
    editor.data = csc_matrix(np.concatenate((editor.data.todense(), new_col), axis=1))
    print("Inserted new column as "+debug_str)
    return

@cli.command('save-svm')
@click.argument('filename', type=click.STRING)
@click.option('--overwrite', is_flag=True)
@pass_editor
def save_svm(editor, filename, overwrite):
    assert editor.data is not None, "no data is loaded"
    path = os.path.join(editor.indir, filename)
    if os.path.exists(path) and not overwrite:
        print(f"Error: '{path}' already exists, use `--overwrite` to save anyway")
        return
    elif os.path.exists(path):
        print(f"Warning: '{path}' already exists, overwriting")
        os.remove(path)
    dump_svmlight_file(editor.data, editor.y, path)
    print(f"Data with shape {editor.data.shape} saved to '{path}'")
    return

@cli.command('split')
@click.argument('dataset_name', type=click.STRING)
@click.option('--K', type=click.INT, default=0, help="if provided, data is only split for K nodes (default: all possible splits)")
@click.option('--seed', type=click.INT, default=None, help="random shuffling seed (default: no shuffle)")
@pass_editor
def split(editor, dataset_name, k, seed):
    if k:
        _split_dataset(editor, dataset_name, k, seed)
        return
    for k in range(1, editor.data.shape[1]+1):
        _split_dataset(editor, dataset_name, k, seed)

def _split_dataset(editor, dataset, K, seed):
    import joblib
    _, n_features = editor.data.shape
    output_folder = os.path.join(editor.outdir, dataset, 'features', str(K))
    os.makedirs(os.path.join(output_folder, 'X'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'y'), exist_ok=True)

    indices = np.arange(n_features)
    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(indices)
    block_size = int(np.ceil(n_features / K))

    beg = 0
    for k in range(K):
        file_x = os.path.join(output_folder, 'X', str(k))
        file_y = os.path.join(output_folder, 'y', str(k))
        joblib.dump(editor.data[:, beg:beg+block_size], file_x)
        joblib.dump(editor.y, file_y)
        beg += block_size

    joblib.dump(indices, os.path.join(output_folder, 'indices'))
    print(f"Splits for world_size {K} stored in '{output_folder}'")


if __name__ == "__main__":
    cli()