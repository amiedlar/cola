from sklearn.datasets import load_svmlight_file,dump_svmlight_file
from sklearn.preprocessing import normalize
from scipy.sparse import csc_matrix
import numpy as np
import click

def center_data(X, axis=0, copy=True):
    m,n = X.shape
    if not axis:
        t = m
        m = n
        n = t
    X_bar = [np.average(X[:,i].todense()) for i in range(m)]
    if copy:
        return np.concatenate([X[:,i].todense() - X_bar[i]*np.ones((n,)) for i in range(len(X_bar))])
    for i in range(m):
        X[:,i] -= X_bar[i]*np.ones((n,))
        return X

@click.command()
@click.option('--input_file', type=click.File, help='path to svm dataset')
@click.option('--axis', type=click.IntRange(0,1), default=0, help='if 0 [default], then centers and normalizes each col, elif 1, each row')
@click.option('--output_file', type=click.File(mode='w'), help='output file path')
def main(input_file, axis, overwrite):
    X, y = load_svmlight_file(input_file)
    X = center_data(X, axis=axis, copy=False)
    normalize(X, axis=axis)

if __name__ == '__main__':
    main()