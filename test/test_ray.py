import ray

@ray.remote(num_gpus=1)
def test_gpu():
    import torch
    return torch.cuda.get_device_name(0)


@ray.remote(num_gpus=1)
def test_cuda():
    import torch
    device = torch.device('cuda' if torch.cuda.
                      is_available() else 'cpu')
    return device

ray.init()
print(ray.get(test_gpu.remote()))
print(ray.get(test_cuda.remote()))
