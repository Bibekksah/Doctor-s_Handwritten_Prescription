import torch


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")


def get_device_name(device):

    if device.type == "cuda":
        return torch.cuda.get_device_name(0)

    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"

    return "CPU"


if __name__ == "__main__":

    device = get_device()

    print("Device:", device)
    print("Device name:", get_device_name(device))