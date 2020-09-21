import torch
from torch import nn, optim
from util import get_device, Logger


class BinaryMixer:
    def __init__(self, A=None, clamp=False, device=None):
        """Mixes two signals x, y using the input matrix A:
            b = x + A.y

        Parameters
        ----------
        A: matrix
            must have number of rows equal to prod(x.shape) and number of
            columns equal to prod(y.shape)
        clamp: bool
            Whether to clamp output to unit interval. default: False
        device: torch.device or str
            cpu or cuda

        """
        self.device = get_device(device)
        self._set_matrix(A)
        self.clamp = clamp

    def _call_dict(self, x, y):
        """ This method is to add support for two measurement matrices.
        """
        if not ((self.A is None) or isinstance(self.A, dict)):
            emsg = f"expected dict for self.A but got {type(self.A)}"
            raise TypeError(emsg)

        # deal with A being None or dict
        if self.A is None:
            A = None
            B = None
        else:
            A = self.A.get("x", None)
            B = self.A.get("y", None)

        # validate shapes of x and y if no random map is used.
        if (A is None) and (B is None):
            if x.shape != y.shape:
                emsg = f"Expected x.shape == y.shape but found {x.shape} != {y.shape}"
                raise ValueError(emsg)

        # Construct mappings
        if A is None:
            Ax = x.to(self.device)
        else:
            Ax = torch.matmul(A, x.view(-1, 1))

        if B is None:
            By = y.to(self.device)
        else:
            By = torch.matmul(B, y.view(-1, 1)).view(*Ax.shape)

        # Compute result
        b = Ax + By
        return b

    def __call__(self, x, y):

        if (self.A is None) or isinstance(self.A, dict):
            b = self._call_dict(x, y)
        else:
            if not isinstance(self.A, torch.Tensor):
                emsg = f"Expected tensor for A but got {type(self.A)}"
                raise TypeError(emsg)
            x = x.to(self.device)
            y = y.to(self.device)
            By = torch.matmul(self.A, y.view(-1, 1)).view(*x.shape)
            b = x + By

        if self.clamp:
            b.clamp_(0.0, 1.0)
        return b

    def _set_matrix(self, A):
        if A is None:
            # no mixing
            self.A = None
            return
        if not isinstance(A, torch.Tensor):
            A = torch.as_tensor(A)
        A = A.to(self.device)
        self.A = A


class NaryMixer:
    def __init__(self, mixing_matrices=None, clamp=False, device=None):
        self.device = get_device(device)
        self._set_matrices(mixing_matrices)
        self.clamp = clamp

    def __call__(self, *args, **kwargs):
        if (len(args) > 0) and (len(kwargs) > 0):
            emsg = "must all be either args or kwargs, not a mix"
            raise ValueError(emsg)

        if len(args) > 0:
            kwargs = {i: arg.to(self.device) for i, arg in enumerate(args)}

        signal_keys = set(kwargs.keys())
        matrix_keys = set(self.A.keys())
        mult_keys = signal_keys.intersection(matrix_keys)
        other_keys = signal_keys.difference(mult_keys)

        if len(other_keys) > 0:
            other_mixed = torch.stack(
                tuple(kwargs[key] for key in other_keys), dim=0
            ).sum(dim=0)
            output_shape = other_mixed.shape
        else:
            output_shape = (1, -1)

        mult = {
            key: torch.matmul(self.A[key], kwargs[key].view(-1, 1)).view(
                *output_shape
            )
            for key in mult_keys
        }
        mult_mixed = torch.stack(
            tuple(img for img in mult.values()), dim=0
        ).sum(dim=0)

        if len(other_keys) > 0:
            mult_mixed = other_mixed + mult_mixed
        if self.clamp:
            mult_mixed.clamp_(0.0, 1.0)
        return mult_mixed

    def _set_matrix(self, key, matrix):
        if (not hasattr(self, key)) or (getattr(self, key) is None):
            self.A = {key: None}

        if matrix is None:
            return
        elif not isinstance(matrix, torch.Tensor):
            matrix = torch.as_tensor(matrix).to(self.device)
            self.A[key] = matrix

    def _set_matrices(self, mixing_matrices):
        if isinstance(mixing_matrices, dict):
            for key, matrix in mixing_matrices.items():
                self._set_matrix(key, matrix)
        elif isinstance(mixing_matrices, (tuple, list)):
            for i, matrix in enumerate(mixing_matrices):
                self._set_matrix(i, matrix)
        else:
            raise TypeError(
                "container of mixing_matrices not recognized; "
                "should be one of dict, tuple list. "
            )


def demixing_problem_one_network(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    A=None,
    num_iter=1000,
    clamp=True,
    device=None,
):
    device = get_device(device)

    mixer = BinaryMixer(A=A, clamp=clamp, device=device)
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


def demixing_problem_two_network(
    networks: dict, imgs: dict, A=None, num_iter=1000, clamp=True, device=None,
):
    """
    Parameters
    ----------
    networks: dict
        dict of networks
    imgs: dict
        dict of images (torch.Tensor objects)
    A: torch.Tensor (optional)
        mixing matrix
    num_iter: int
        Maximum number of iterations
    clamp: bool
        Whether to clip the output to [0, 1].
    device: str or torch.device
        'cpu' or 'cuda' device.

    Returns
    -------
    results: dict
        With keys: 
            demixed0 : demixed image corresponding with networks[list(networks.keys())[0]]
            demixed1 : demixed image corresponding with networks[list(networks.keys())[1]]
            logger : util.Logger object with 'iter' and 'loss' items.
            mixer : The BinaryMixer object used to mix x0 and x1
            mixture : mixer(x0, x1) where mixer = BinaryMixer(A, clamp)
            optimizer : The optimizer is hard-coded to be optim.Adam([w0, w1], lr=1e-2).
            w0 : The latent space parameters such that net0(w0) = demixed0
            w1 : The latent space parameters such that net1(w1) = demixed1
            x0 : The ground-truth image x0
            x1 : The ground-truth image x1
    """
    device = get_device(device)

    mixer = BinaryMixer(A=A, clamp=clamp, device=device)
    img_classes = list(imgs.keys())
    if len(img_classes) != 2:
        emsg = f"Expected binary demixing problem but found {len(img_classes)} images."
        raise ValueError(emsg)
    x0 = imgs[img_classes[0]]
    x1 = imgs[img_classes[1]]
    mixture = mixer(x0, x1)

    net_keys = list(networks.keys())
    net0 = networks[net_keys[0]].eval().to(device)
    net1 = networks[net_keys[1]].eval().to(device)

    # get parameters in latent space of each generator
    if mixture.ndim == 4:
        p0 = net0.encode(mixture)
        p1 = net1.encode(mixture)
    elif mixture.ndim == 3:
        p0 = net0.encode(mixture.unsqueeze_(0))
        p1 = net1.encode(mixture)
    else:
        p0 = net0.encode(mixture.view(1, -1))
        p1 = net1.encode(mixture.view(1, -1))

    mixture_enc0 = net0.reparametrize(*p0)
    mixture_enc1 = net1.reparametrize(*p1)

    # Set requires_grad = False for all model parameters.
    net0.requires_grad_(False)
    net1.requires_grad_(False)

    # For encoding vectors w0 and w1, set requires_grad = True.
    w0 = (
        mixture_enc0.clone()
        .detach()
        .add_(torch.randn_like(mixture_enc0), alpha=0.1)
        .requires_grad_(True)
    )
    w1 = (
        mixture_enc1.clone()
        .detach()
        .add_(torch.randn_like(mixture_enc1), alpha=0.1)
        .requires_grad_(True)
    )

    # Acquire im0 = net0.decode(w0), im1 = net1.decode(w1)
    #   and compute loss = norm(y - mixer(im0, im1), 2)**2 where
    #   w0.requires_grad = True and w1.requires_grad = True.
    # Then after we call loss.backward, we should have updates for
    #   w0 and w1. Just gotta' pass w0 and w1 to the optimizer.

    criterion = nn.MSELoss()
    optimizer = optim.Adam([w0, w1], lr=1e-2)
    logger = Logger()

    for i in range(num_iter):
        demixed0 = net0.decode(w0)
        demixed1 = net1.decode(w1)
        optimizer.zero_grad()
        mixture_pred = mixer(demixed0, demixed1)
        loss = criterion(mixture_pred.view(1, -1), mixture.view(1, -1))
        logger("iter", i)
        logger("loss", loss.item())
        loss.backward()
        optimizer.step()

    results = {
        "demixed0": demixed0,
        "demixed1": demixed1,
        "logger": logger,
        "mixer": mixer,
        "mixture": mixer(x0, x1),
        "mixture_pred": mixture_pred,
        "optimizer": optimizer,
        "w0": w0,
        "w1": w1,
        "x0": x0,
        "x1": x1,
    }
    return results


def multi_demixing_problem_one_network(
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
