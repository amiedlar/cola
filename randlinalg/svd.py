from numpy.random import randn, seed
from numpy.linalg import qr, svd

def randomized_range_finder(B, l, random_state=None):
    seed(random_state)
    R = randn(B.shape[1], l)
    Y = B * R
    Q, _ = qr(Y)
    return Q

def rsvd(B, l, threshold=0.0, random_state=None):
    Q = randomized_range_finder(B, l, random_state=random_state)
    C = Q.H * B
    U, s, Vh = svd(C, full_matrices=False)
    indices = s > threshold
    U = U[:,indices]
    s = s[indices]
    Vh = Vh[indices,:]
    U = Q * U
    return U, s, Vh
