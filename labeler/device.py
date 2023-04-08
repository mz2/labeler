from torch import device
from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available

def get_device() -> device:
    if cuda_is_available():
        return device("cuda")
    elif mps_is_available():
        return device("mps")
    else:
        return device("cpu")