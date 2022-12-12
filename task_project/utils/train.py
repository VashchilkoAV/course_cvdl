import warnings

import torch
from torch import optim


def train(dataset, net=None, criterion=None, optimizer=None, batch_size=8, lr=3e-4, epochs=20, device=None):
    log = []
    if device is not None:
        net.to(device)
    
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=lr)

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    stats_step = (len(dataset) // 10 // batch_size) + 1
    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss).any():
                warnings.warn("nan loss! skip update")
                print(f"last loss: {loss.item()}")
                break
            running_loss += loss.item()
            if (i % stats_step == 0):
                print(f"epoch {epoch}|{i}; total loss:{running_loss / stats_step}")
                print(f"last losses: {loss.item()}")
                log.append(loss.item())
                running_loss = 0.0
            loss.backward()
            optimizer.step()
    print('Finished Training')
    return net, log