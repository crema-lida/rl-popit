import torch
import torch.nn as nn
import torch.nn.functional as f


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        features = 64
        self.conv_block = nn.Sequential(
            nn.Conv2d(4, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.Tanh()
        )
        self.res_net = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(features, features, 3, padding=1),
                    nn.BatchNorm2d(features),
                    nn.Tanh(),
                    nn.Conv2d(features, features, 3, padding=1),
                    nn.BatchNorm2d(features),
                ) for _ in range(1)
            ]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(features, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(72, 36),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(features, 1, 1),
            nn.BatchNorm2d(1),
            nn.Softplus(),
            nn.Flatten(),
            nn.Linear(36, features),
            nn.Softplus(),
            nn.Linear(features, 1),
            nn.Tanh(),
        )
        self.policy_network = [self.conv_block, self.res_net, self.policy_head]

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_net:
            residual = block(x)
            x = torch.tanh(residual)
        return x

    def forward_policy_head(self, x, mask):
        policy = self.policy_head(x).masked_fill(mask, -torch.inf)
        return f.softmax(policy, dim=1)

    def forward_value_head(self, x):
        return self.value_head(x)


if __name__ == '__main__':
    import torch.onnx as onnx

    x = torch.zeros(128, 4, 6, 6, dtype=torch.float)
    onnx.export(Network(), x, './model.onnx')
