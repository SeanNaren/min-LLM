import time

import matplotlib.pyplot as plt
import tinycudann as tcnn
import torch
from torch import autocast
from tqdm import tqdm


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


def benchmark(cls, dim_model, input, targets, num_warmup_steps, num_steps, device, should_autocast):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    model = cls(dim_model)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    print(f"Testing {cls.__name__}, Number of parameters: {sum(p.numel() for p in model.parameters())}")
    for x in range(num_warmup_steps):
        with autocast(device_type='cuda', enabled=should_autocast):
            output = model(input)
            loss = criterion(output, targets)
        loss.backward()

    torch.cuda.synchronize()
    start = time.time()
    for x in range(num_steps):
        with autocast(device_type='cuda', enabled=should_autocast):
            output = model(input)
            loss = criterion(output, targets)
        loss.backward()
    torch.cuda.synchronize()
    end = time.time()
    memory = torch.cuda.max_memory_allocated() / 2 ** 20
    return end - start, memory


def run_compare(dim_model, should_autocast):
    num_steps = 100
    num_warmup_steps = 20
    batch_size = 2
    seq_length = 2048

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    input = torch.rand([batch_size, seq_length, dim_model], dtype=torch.float32).to(device)
    targets = torch.rand([batch_size, seq_length, dim_model], dtype=torch.float32).to(device)

    vanilla_time, vanilla_memory = benchmark(
        cls=MLP,
        dim_model=dim_model,
        input=input,
        targets=targets,
        device=device,
        num_warmup_steps=num_warmup_steps,
        num_steps=num_steps,
        should_autocast=should_autocast
    )
    tqdm.write(f"Time taken to benchmark Vanilla: {vanilla_time:.4f}s Memory used: {vanilla_memory:.4f}MB")

    tiny_time, tiny_memory = benchmark(
        cls=TinyMLP,
        dim_model=dim_model,
        input=input,
        targets=targets,
        device=device,
        num_warmup_steps=num_warmup_steps,
        num_steps=num_steps,
        should_autocast=should_autocast
    )
    tqdm.write(f"Time taken to benchmark Tiny: {tiny_time:.4f}s Memory used: {tiny_memory:.4f}MB")
    return (vanilla_time, vanilla_memory), (tiny_time, tiny_memory)


if __name__ == '__main__':

    should_autocast = False

    dims = [128, 512, 1024, 2048, 4096, 8192, 16384]

    vanilla_records, tiny_records = [], []
    for dim_model in tqdm(dims, total=len(dims)):
        vanilla, tiny = run_compare(dim_model, should_autocast)
        vanilla_records.append(vanilla)
        tiny_records.append(tiny)

    vanilla_time = [vanilla[0] for vanilla in vanilla_records]
    tiny_time = [tiny[0] for tiny in tiny_records]
    vanilla_mem = [vanilla[1] for vanilla in vanilla_records]
    tiny_mem = [tiny[1] for tiny in tiny_records]

    plt.plot(dims, vanilla_time, label="Vanilla")
    plt.plot(dims, tiny_time, label="Tiny")
    plt.xlabel("Dimension of MLP")
    plt.ylabel("Log Scale Measured Time (s) (lower is better)")
    plt.yscale('log')
    plt.legend()
    plt.savefig("measured_time.png")

    plt.clf()
    plt.plot(dims, vanilla_mem, label="Vanilla")
    plt.plot(dims, tiny_mem, label="Tiny")
    plt.xlabel("Dimension of MLP")
    plt.ylabel("Log Scale Measured Memory (MB) (lower is better)")
    plt.yscale('log')
    plt.legend()
    plt.savefig("measured_mem.png")
