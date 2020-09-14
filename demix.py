import torch
from torch import nn, optim
from util import Logger


def demixing_problem(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    Q=None,
    num_iter=1000,
    clamp=True,
):

    x0_shape = x0.shape
    x1_shape = x1.shape
    assert (
        x0_shape == x1_shape
    ), f"Expected x0.shape == x1.shape but found {x0.shape} != {x1.shape}"
    if (Q is not None) and isinstance(Q, torch.Tensor):
        mixture = x0 + torch.matmul(Q, x1.view(-1, 1)).view(*x1_shape)
    else:
        mixture = x0 + x1

    if clamp:
        mixture.clamp_(0.0, 1.0)

    model.eval()

    mixture_params = model.encode(mixture.view(1, -1))
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
