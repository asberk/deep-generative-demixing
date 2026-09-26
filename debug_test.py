import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(2020)

X_tens = torch.rand((100, 1, 16, 16)).float()
y_tens = torch.randint(0, 2, size=(100,)).long()
X_tens[y_tens == 1] = X_tens[y_tens == 1] + 1
dset = TensorDataset(X_tens, y_tens)
dloader = DataLoader(dset, batch_size=8, shuffle=True)


layer_weights = []


class MyNet(nn.Module):
    def __init__(self, parm=0):
        super().__init__()

        self.layer1 = nn.Conv2d(1, 2, (3, 3))
        self.pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        # self.layer2 = nn.Conv2d(2, 1, (3, 3))
        # self.layer3 = nn.Linear(4, 2)
        if parm == 0:
            self.layer2 = nn.Conv2d(2, 1, (3, 3))
            self.layer3 = nn.Linear(4, 2)
        elif parm == 1:
            self.layer2 = nn.Conv2d(2, 2, (3, 3))
            self.layer3 = nn.Linear(8, 2)

    def forward(self, input):
        layer_weights.append(
            [
                float(torch.norm(xx.weight.detach()).numpy())
                for xx in [self.layer1, self.layer2, self.layer3]
            ]
        )
        out = self.relu(self.pool(self.layer1(input)))
        out = self.relu(self.pool(self.layer2(out)))
        out = self.layer3(out.view(out.shape[0], -1))
        return out


def test():
    torch.manual_seed(2020)
    my_net0 = MyNet(parm=0)
    my_net1 = MyNet(parm=1)

    for batch in dloader:
        break

    imgs = batch[0]
    print("\nparm=0")
    out0 = my_net0(imgs)
    print("\nparm=1")
    out1 = my_net1(imgs)
    return


def main():
    torch.manual_seed(2020)
    my_net = MyNet(1)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.SGD(my_net.parameters(), lr=1e-4)

    for epoch in range(2):
        for batch in dloader:
            optimizer.zero_grad()
            imgs, targs = batch
            raw_logits = my_net(imgs)
            loss = criterion(raw_logits, targs)
            loss.backward()
            optimizer.step()

    for item in layer_weights:
        print(item)


if __name__ == "__main__":
    # print("\nTest")
    # test()

    print("\nMain")
    main()
