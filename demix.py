import torch
from torch import nn, optim
from util import get_device, Logger


def BinaryMixer:
    def __init__(self, A=None, clamp=False, device=None):
        """Mixes two signals x, y using the input matrix A:
            b = x + A.y

        Parameters
        ----------
        A: matrix
            must have number of rows equal to prod(x.shape) and number of columns equal to prod(y.shape)
        clamp: bool
            Whether to clamp output to unit interval. default: False
        device: torch.device or str
            cpu or cuda

        """
        device = get_device(device)

        if A is None:
            # no mixing
            self.A = None
        elif not isinstance(A, torch.Tensor):
            A = torch.as_tensor(A).to(device)
            self.A = A

        self.clamp = clamp
        self.device = device

    def __call__(self, x, y):
        x_shape = x.shape
        y_shape = y.shape

        if self.A is None:
            assert (x_shape == y_shape), f"Expected x.shape == y.shape but found {x.shape} != {y.shape}"
            b = (x + y).to(self.device)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

            b = x + torch.matmul(self.A, y.view(-1, 1)).view(*x_shape)

        if self.clamp:
            b.clamp_(0., 1.)
        return b

def demixing_problem(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    Q=None,
    num_iter=1000,
    clamp=True,
    device=None,
):
    device = get_device(device)

    mixer = BinaryMixer(A=Q, clamp=clamp, device=device)
    mixture = mixer(x0, x1)

    model = model.eval().to(device)

    try:
        mixture_params = model.encode(mixture.view(1, -1))
    except:
        mixture_params = model.encode(mixture.unsqueeze_(0))

    mixture_encoding = model.reparametrize(*mixture_params)

    # Set requires_grad = False for all model parameters.
    model.requires_grad_(False)

    # For encoding vectors w0 and w1, set requires_grad = True.
    w0 = (
        mixture_encoding.clone()
        .detach()
        .add_(torch.randn_like(mixture_encoding), alpha=0.1)
        .requires_grad_(True)
    )

    w1 = (
        mixture_encoding.clone()
        .detach()
        .add_(torch.randn_like(mixture_encoding), alpha=0.1)
        .requires_grad_(True)
    )

    # Acquire im0 = model.decode(w0), im1 = model.decode(w1)
    #   and compute loss = norm(y - im0 - im1, 2)**2 where
    #   w0.requires_grad = True and w1.requires_grad = True.
    # Then after we call loss.backward, we should have updates for
    #   w0 and w1. Just gotta' pass w0 and w1 to the optimizer.

    criterion = nn.MSELoss()
    optimizer = optim.Adam([w0, w1], lr=1e-2)
    logger = Logger()

    for i in range(num_iter):
        demixed0 = model.decode(w0)
        demixed1 = model.decode(w1)
        optimizer.zero_grad()
        mixture_pred = (demixed0 + demixed1).squeeze()
        loss = criterion(mixture_pred, mixture)
        logger("iter", i)
        logger("loss", loss.item())
        loss.backward()
        optimizer.step()
    return (
        demixed0,
        w0,
        demixed1,
        w1,
        mixture,
        mixture_encoding,
        optimizer,
        logger,
    )


def multi_demixing_problem(
    model: nn.Module, images: dict, num_iter=1000, clamp=True, verbose=True
):

    img_shapes = [img.shape for img in images.values()]
    assert all(
        shape0 == shape1
        for i, shape0 in enumerate(img_shapes)
        for shape1 in img_shapes[i:]
    ), "Shape mismatch."

    mixture = torch.stack(tuple(images.values()), dim=0).sum(dim=0)

    if clamp:
        mixture.clamp_(0.0, 1.0)

    model.eval()

    mixture_params = model.encode(mixture.view(1, -1))
    mixture_encoding = model.reparametrize(*mixture_params)

    # Set requires_grad = False for all model parameters.
    model.requires_grad_(False)

    # For encoding vectors w0 and w1, set requires_grad = True.

    Wi = [
        mixture_encoding.clone()
        .detach()
        .add_(torch.randn_like(mixture_encoding), alpha=0.1)
        .requires_grad_(True)
        for _ in range(len(images))
    ]

    # Acquire im0 = model.decode(w0), im1 = model.decode(w1)
    #   and compute loss = norm(y - im0 - im1, 2)**2 where
    #   w0.requires_grad = True and w1.requires_grad = True.
    # Then after we call loss.backward, we should have updates for
    #   w0 and w1. Just gotta' pass w0 and w1 to the optimizer.

    criterion = nn.MSELoss()
    optimizer = optim.Adam(Wi, lr=1e-2)
    logger = Logger()

    for i in range(num_iter):
        demixed = [model.decode(w) for w in Wi]
        optimizer.zero_grad()
        mixture_pred = torch.stack(demixed, dim=0).sum(dim=0).squeeze()
        loss = criterion(mixture_pred, mixture)
        logger("iter", i)
        logger("loss", loss.item())
        if verbose and ((i % 100) == 0):
            i_string = f"{i}"
            print(f"iter {i_string:5s} loss {loss.item():.4f}")
        loss.backward()
        optimizer.step()
    return demixed, Wi, mixture, mixture_encoding, optimizer, logger
