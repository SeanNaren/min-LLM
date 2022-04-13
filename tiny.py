import time

import tinycudann as tcnn
import torch


class TinyMLP(torch.nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()
        config = {
            "otype": "CutlassMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": dim_model,
            "n_hidden_layers": 1
        }
        self.dim_model = dim_model

        self.mlp = tcnn.Network(self.dim_model, dim_model, config)

    def forward(self, x):
        x = x.view(-1, self.dim_model)  # collapse b & t [b&t,d]
        x = self.mlp(x)
        x = x.view(2, -1, self.dim_model)  # un-collapse b & t [b,t,d]?
        return x


class MLP(torch.nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()
        self.dim_model = dim_model

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_model, dim_model, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_model, dim_model, bias=False)
        )

    def forward(self, x):
        return self.mlp(x)


def benchmark(cls, dim_model, input, num_warmup_steps, num_steps, device):
    model = cls(dim_model)
    model.to(device)
    print(f"Testing {cls.__name__}, Number of parameters: {sum(p.numel() for p in model.parameters())}")
    torch.cuda.synchronize()
    for x in range(num_warmup_steps):
        model(input)

    torch.cuda.synchronize()
    start = time.time()
    for x in range(num_steps):
        model(input)
    torch.cuda.synchronize()
    end = time.time()
    return end - start


if __name__ == '__main__':
    # kinda yolo, saw 8192 for chinchilla (deepminds 70B param model)
    # took a very rough /7 (to estimate a model size of 10B params)
    dim_model = 1024 * 4
    num_steps = 500
    num_warmup_steps = 100

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    input = torch.rand([2, 1024, dim_model], dtype=torch.float32).to(device)

    measured_time = benchmark(
        cls=MLP,
        dim_model=dim_model,
        input=input,
        device=device,
        num_warmup_steps=num_warmup_steps,
        num_steps=num_steps
    )
    print(f"Time taken to benchmark Vanilla: {measured_time:.2f}s")

    measured_time = benchmark(
        cls=TinyMLP,
        dim_model=dim_model,
        input=input,
        device=device,
        num_warmup_steps=num_warmup_steps,
        num_steps=num_steps
    )
    print(f"Time taken to benchmark Tiny: {measured_time:.2f}s")
