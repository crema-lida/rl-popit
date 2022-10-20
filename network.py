import torch
import torch.nn as nn
import torch.nn.functional as f


class CNN(nn.Module):
    def __init__(self, in_features=2, num_blocks=2):
        super(CNN, self).__init__()
        features = 64
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, features, 3, padding=1),
            # nn.BatchNorm2d(features),
            nn.Tanh(),
        )
        self.middle_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(features, features, 3, padding=1),
                    # nn.BatchNorm2d(features),
                    nn.Tanh(),
                ) for _ in range(num_blocks)
            ]
        )
        self.policy_head = PolicyHead(features)
        self.value_head = ValueHead(features)

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.middle_layers:
            x = block(x)
        return x

    @property
    def policy_network(self):
        return [self.conv_block, self.middle_layers, self.policy_head]


class ResNet(CNN):
    def __init__(self, in_features=2, num_blocks=3):
        super(ResNet, self).__init__(in_features)
        features = 64
        self.middle_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(features, features, 3, padding=1),
                    nn.BatchNorm2d(features),
                    nn.Tanh(),
                    nn.Conv2d(features, features, 3, padding=1),
                    nn.BatchNorm2d(features),
                ) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.middle_layers:
            x = torch.tanh(x + block(x))
        return x


class PolicyHead(nn.Module):
    def __init__(self, features):
        super(PolicyHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(features, 2, 1),
            # nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(72, 36),
        )

    def forward(self, x, mask):
        x = self.layers(x).masked_fill(mask, -torch.inf)
        return f.softmax(x, dim=1)


class ValueHead(nn.Module):
    def __init__(self, features):
        super(ValueHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(features, 1, 1),
            nn.BatchNorm2d(1),
            nn.Softplus(),
            nn.Flatten(),
            nn.Linear(36, features),
            nn.Softplus(),
            nn.Linear(features, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
    cnn = CNN()
    x = torch.zeros(128, 2, 6, 6, dtype=torch.float)
    mask = x[:, 1].reshape(-1, 36) != 0
    writer.add_graph(cnn, x)
    writer.close()
