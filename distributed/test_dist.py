import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def train(rank, world_size):
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=rank,
                            world_size=world_size)

    acc_steps = 4
    eval_steps = 8
    device = torch.device('cuda', rank)
    model = nn.Linear(10, 10).to(device)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    for i in range(1, 24 + 1):
        print(f'epoch={i}, rank={rank}')
        # forward
        outputs = ddp_model(torch.randn(20, 10).to(device))
        labels = torch.randn(20, 10).to(device)
        # backward
        loss_fn(outputs, labels).backward()

        # update with accumulation
        if i % acc_steps == 0:
            print(f'epoch={i}, rank={rank}, step')
            optimizer.step()
            optimizer.zero_grad()
            # run evaluation (only forward)
            if rank == 0 and i % eval_steps == 0:
                with torch.no_grad():
                    for _ in range(10000):  # Should take long time
                        outputs = model(torch.rand(20, 10).to(device))
                    print(f'epoch={i}, rank={rank}, evaluation')


def main():
    n_gpu = torch.cuda.device_count()
    mp.spawn(train,
             nprocs=n_gpu,
             args=(n_gpu,),
             join=True)


if __name__ == '__main__':
    main()
