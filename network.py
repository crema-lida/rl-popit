import torch
import torch.nn as nn
import torch.nn.functional as f


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        features = 12
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.Softplus()
        )
        self.res_net = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(features, features, 3, padding=1),
                    nn.BatchNorm2d(features),
                    nn.Softplus(),
                    nn.Conv2d(features, features, 3, padding=1),
                    nn.BatchNorm2d(features),
                )
            ] * 4
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(features, 2, 1),
            nn.BatchNorm2d(2),
            nn.Softplus(),
            nn.Flatten(),
            nn.Linear(72, 36)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(features, 1, 1),
            nn.BatchNorm2d(1),
            nn.Softplus(),
            nn.Flatten(),
            nn.Linear(36, features),
            nn.Softplus(),
            nn.Linear(features, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_net:
            residual = block(x)
            x = f.softplus(x + residual)
        return x

    def forward_policy_head(self, x, mask):
        policy = self.policy_head(x)
        policy.masked_fill_(mask, -torch.inf)
        return f.softmax(policy, dim=1)

    def forward_value_head(self, x):
        return self.value_head(x)


if __name__ == '__main__':
    import torch.onnx as onnx

    x = torch.zeros(128, 3, 6, 6, dtype=torch.float)
    onnx.export(Network(), x, './model.onnx')
