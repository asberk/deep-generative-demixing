import numpy as np
import matplotlib.pyplot as plt


def relu(input):
    """
    relu(input)

    relu(x) = max(0, x); acts element-wise.

    Parameters
    ----------
    input: np.ndarray
        input vector

    Returns
    -------
    output: np.ndarray
        ReLU-activated vector
    """
    return np.clip(input, 0, np.inf)


def neural_net(d0=1, dn=5, with_bias=True, num_points=501, seed=None):
    """
    neural_net(d0=1, dn=5, with_bias=True, num_points=501, seed=None)

    Computes the output from a generative neural net with random normal weight
    matrices of size (d+1, d).

    Parameters
    ----------
    d0: int
        description
    dn: int
        description
    with_bias: bool
        
    num_points: int
        number of points to use along each domain axis
    seed: int
        description

    """
    # t = np.linspace(-20, 20, num_points)
    # T = np.stack(np.meshgrid(*[t for _ in range(d0)])).reshape(d0, -1)
    T = np.random.randn(d0, num_points)

    A = [np.random.randn(i + 1, i) for i in range(d0, dn)]
    if with_bias:
        b = [np.random.rand(i + 1, 1) for i in range(d0, dn)]

    c = T.prod(axis=0)
    out = T.copy()
    for i in range(dn - d0):
        out = A[i].dot(out)
        if with_bias:
            out = out + b[i]
        if i < dn - 1:
            out = relu(out)
    return out, c


def make_a_plot(d0=1, dn=5, with_bias=True, num_points=501, seed=None):

    xy, c = neural_net(d0, dn, with_bias, num_points, seed)

    ell = xy[0].size
    J = np.random.choice(ell, size=num_points, replace=False)

    fig, ax = plt.subplots(dn, dn, figsize=(9, 8))
    for i in range(dn):
        for j in range(dn):
            if i == j:
                ax[i, j].axis("off")
                continue
            ax[i, j].vlines(0, -10, 10, linestyle="dotted")
            ax[i, j].hlines(0, -10, 10, linestyle="dotted")
            ax[i, j].scatter(xy[i][J], xy[j][J], c=c[J], s=2)
            ax[i, j].set_xlim(-3, 3)
            ax[i, j].set_ylim(-3, 3)

    plt.tight_layout()
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    seed = 1234
    with_bias = True
    num_points = 5001
    make_a_plot(d0=2, with_bias=with_bias, num_points=num_points, seed=seed)

    make_a_plot(
        d0=2, dn=6, with_bias=with_bias, num_points=num_points, seed=seed
    )
