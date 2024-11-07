import subprocess


def select_cuda_device(min_free_mb=11000):
    available_devices = []

    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
        text=True,
    )

    free_memory = [int(x) for x in nvidia_smi.stdout.strip().split("\n")]
    for i, mem_free in enumerate(free_memory):
        if mem_free >= min_free_mb:
            available_devices.append((i, mem_free))

    if available_devices:
        best_available = max(available_devices, key=lambda x: x[1])
        device = best_available[0]
        mb = best_available[1]
        max(available_devices, key=lambda x: x[1])[1]
        print(f"Selected device {device}, which has {mb} MB available.")
        return device
    else:
        raise RuntimeError("All devices are occupied.")
