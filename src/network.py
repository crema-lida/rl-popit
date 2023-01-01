import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, features=64, num_blocks=2):
        super(CNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(2, features, 3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.Tanh(),
        )
        self.middle_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(features, features, 3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
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
    def __init__(self, features=64, num_blocks=3):
        super(ResNet, self).__init__(features)
        self.middle_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(features, features, 3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.Tanh(),
                    nn.Conv2d(features, features, 3, padding=1, bias=False),
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
            nn.Conv2d(features, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(72, 36),
        )

    def forward(self, x):
        return self.layers(x)


class ValueHead(nn.Module):
    def __init__(self, features):
        super(ValueHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(features, 1, 1, bias=False),
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
    import os
    import onnxruntime
    import numpy as np

    MODEL_NAME = 'cnn2-64'
    EXPORT_DIR = f'../best_models/{MODEL_NAME}'
    os.makedirs(EXPORT_DIR, exist_ok=True)

    with torch.no_grad():
        # save model structure and its params
        model = CNN(features=64, num_blocks=2).eval()
        model.load_state_dict(torch.load(f'../{MODEL_NAME}/checkpoint')['model'])

        # prepare input for model structure inference and testing results
        state = torch.randint(0, 2, (1, 2, 6, 6), dtype=torch.float32)
        out = model(state)
        policy = model.policy_head(out)
        value = model.value_head(out)

        inputs = [
            {'state': state},
            {'conv.out': out},
            {'conv.out': out}
        ]
        torch_model = [model, model.policy_head, model.value_head]
        onnx_model = {}
        for i, module in enumerate(['conv_block', 'policy_head', 'value_head']):
            torch.onnx.export(torch_model[i], tuple(inputs[i].values()), f'{EXPORT_DIR}/{module}.onnx',
                              input_names=list(inputs[i].keys()))
            onnx_model[module] = onnxruntime.InferenceSession(f'{EXPORT_DIR}/{module}.onnx')

        # compare ONNX Runtime and PyTorch results
        onnx_out = onnx_model['conv_block'].run(None, {'state': state.numpy()})[0]
        onnx_policy = onnx_model['policy_head'].run(None, {'conv.out': onnx_out})[0]
        onnx_value = onnx_model['value_head'].run(None, {'conv.out': onnx_out})[0]
        if np.allclose(out.numpy(), onnx_out, rtol=1e-3, atol=1e-5) and \
                np.allclose(policy, onnx_policy, rtol=1e-3, atol=1e-5) and \
                np.allclose(value, onnx_value, rtol=1e-3, atol=1e-5):
            print('Exported model has been tested with ONNXRuntime, and the result looks good!')
        else:
            raise Exception('Model exported but the test failed!')
