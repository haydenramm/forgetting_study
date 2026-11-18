"""
Startup script to check whether CUDA is available, and if so, what
is available to us in the environment.
"""

import torch


def elucidate_cuda():
    """If CUDA is found, explore the environment."""
    cuda_count = torch.cuda.device_count()
    print("====CUDA ENV====")
    print(f"No devices: {cuda_count}")
    for i in range(cuda_count):
        print(f"{i}", torch.cuda.get_device_properties(i).name)
    print("================")


def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available.")
    else:
        print("CUDA available")
        elucidate_cuda()


if __name__ == "__main__":
    main()
