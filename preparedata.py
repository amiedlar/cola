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
@click.option('--scale-by', default=1, help="scale factor for vector specified by `--scale-col` (default 1)")
@click.option('--weights', type=click.STRING, default=None,
    help="string containing python array with length n_col."
         "values in array correspond weights of each remaining column for replacement linear combination.")
@pass_editor
def replace_column(editor, col, scheme, scale_col, scale_by, weights):
    print('entered replace')
    assert editor.data is not None, "load data before attempting to edit"
    assert weights is None and scheme == 'weights', "specify weighting scheme"
    
    n_row, n_col = editor.data.shape
    if scheme == 'weights':
        weights = np.loads(weights)
    elif scheme == 'scale':
        weights = np.zeros((n_col-1,))
        weights[scale_col] = scale_by
    elif scheme == 'uniform':
        weights = np.array([1/(n_col-1)]*(n_col-1))
    else:
        return NotImplementedError

    weights = np.insert(weights, col, 0)

    new_col = editor.data * weights
    from scipy.sparse import csc_matrix
    editor.data[:,col] = csc_matrix(new_col.reshape((n_row,1)))
    return

@cli.command('insert-columns')
@click.argument('n', type=click.INT)
@click.option('--weights', type=click.STRING, default=None)
@pass_editor
def insert_columns(editor, n, weights):
    print('entered replace')
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
@click.option('--scale-by', default=1, help="scale factor for vector specified by `--scale-col` (default 1)")
@click.option('--weights', type=click.STRING, default=None,
    help="string containing python array with length n_col."
         "values in array correspond weights of each remaining column for replacement linear combination.")
@pass_editor
def insert_column(editor, scheme, scale_col, scale_by, weights):
    _insert_column(editor, scheme, scale_col, scale_by, weights)

def _insert_column(editor, scheme, scale, scale_by, weights):
    assert editor.data is not None, "load data before attempting to edit"
    assert weights is None and scheme == 'weights', "specify weighting scheme"
    
    n_row, n_col = editor.data.shape

    if scheme == 'weights':
        weights = np.loads(weights)
    elif scheme == 'scale':
        weights = np.zeros((n_col,))
        weights[scale] = scale_by
    elif scheme == 'uniform':
        weights = np.array([1/n_col]*n_col)
    else:
        return NotImplementedError

    weights = weights.reshape((n_row,1))
    
    from scipy.sparse import csc_matrix
    new_col = csc_matrix(editor.data * weights)

    editor.data = np.concatenate(editor.data, new_col)
    print("Column inserted")
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
    print(f"Data saved to '{path}'")
    dump_svmlight_file(editor.data, editor.y, path)
    return

if __name__ == "__main__":
    cli()