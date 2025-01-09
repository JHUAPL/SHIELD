# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from typing import Union, Tuple, List


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.square(x)


def moment(x: torch.Tensor, std: float, mean: float, deg: int = 4, eps: float = 1e-4) -> torch.Tensor:
    N = x.shape[0]
    return (1.0 / N) * torch.sum((x - mean) ** deg) / (std ** deg + eps)


def activation_helper(activation: str = 'gelu',
                      gelu_degree: int = 16):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'polygelu':
        raise ValueError("Not supported.")
    elif activation == 'square':
        return Square()
    else:
        return nn.ReLU()


def conv_block(in_ch: int,
               out_ch: int,
               activation: str = 'relu',
               gelu_degree: int = 16,
               pool: bool = False,
               pool_method: str = 'avg',
               kernel: int = 3,
               stride: int = 1,
               padding: Union[int, str] = 1):
    layers = [nn.Conv2d(in_ch,
                        out_ch,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding),
              nn.BatchNorm2d(out_ch),
              activation_helper(activation, gelu_degree)
              ]
    if pool:
        layers.append(nn.MaxPool2d(2, 2) if pool_method == 'max' else nn.AvgPool2d(2, 2))
    return nn.Sequential(*layers)


def get_small_model_dict(activation='gelu',
                         gelu_degree: int = 16,
                         pool_method: str = 'avg') -> nn.ModuleDict:
    classifier = nn.Sequential(nn.Flatten(),
                               nn.Linear(8 * 8 * 128, 10))
    return nn.ModuleDict(
        {
            "conv1": conv_block(in_ch=1,
                                out_ch=64,
                                kernel=4,
                                pool=True,
                                pool_method=pool_method,
                                padding='same',
                                activation=activation,
                                gelu_degree=gelu_degree),
            "conv2": conv_block(in_ch=64,
                                out_ch=128,
                                kernel=4,
                                pool=True,
                                pool_method=pool_method,
                                padding='same',
                                activation=activation,
                                gelu_degree=gelu_degree),
            "conv3": conv_block(in_ch=128,
                                out_ch=128,
                                kernel=4,
                                pool=False,
                                padding='same',
                                activation=activation,
                                gelu_degree=gelu_degree),
            "classifier": classifier
        }
    )


class SmallModel(nn.Module):

    def __init__(self, activation='gelu', gelu_degree: int = 16, pool_method: str = 'avg'):
        super(SmallModel, self).__init__()
        self.model_layers = get_small_model_dict(activation=activation, gelu_degree=gelu_degree,
                                                 pool_method=pool_method)
        self.n_bn_classes = self.count_instances_of_a_class()

    def count_instances_of_a_class(self, cls: nn.BatchNorm2d = nn.BatchNorm2d) -> int:
        n_classes = 0
        for _, block in self.model_layers.items():
            for layer in block:
                # Handle the nested case
                if isinstance(layer, nn.Sequential):
                    for sublayer in layer:
                        if isinstance(sublayer, cls):
                            n_classes += 1
                # Handle the unnested case
                else:
                    if isinstance(layer, cls):
                        n_classes += 1
        return n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.bn_outputs = {}  # key=layer name, v=list of torch.Tensors
        self.outputs = {}

        for name, block in self.model_layers.items():
            block_output, block_bn_output = self.block_pass(block, x)
            self.bn_outputs[name] = block_bn_output

            # Residual Connection
            if "res" in name:
                x = x + block_output
            # Normal
            else:
                x = block_output

        return x

    def block_pass(self, block: nn.Sequential, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        bn_output = []

        # Iterate through a block, which may be nested (residual connections are nested)
        for layer in block:
            # Handle the nested case
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    x = sublayer(x)
                    if isinstance(sublayer, nn.BatchNorm2d):
                        bn_output.append(x)

            # Handlle the unnested case
            else:
                x = layer(x)
                if isinstance(layer, nn.BatchNorm2d):
                    bn_output.append(x)

            self.outputs[layer] = x

        return x, bn_output

    # Must be called after forward method to set self.bn_outputs
    def get_bn_loss_metrics(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means, stds, skews, kurts = self.get_moments_by_layer()

        # Aggregating
        loss_means = F.mse_loss(means, torch.zeros(self.n_bn_classes))
        loss_stds = F.mse_loss(stds, torch.ones(self.n_bn_classes))
        # loss_skews = F.mse_loss(skews, torch.zeros(self.n_bn_classes))
        loss_kurts = F.mse_loss(kurts, 3 * torch.ones(self.n_bn_classes))
        return loss_means, loss_stds, loss_kurts

    def get_moments_by_layer(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means, stds = torch.zeros(self.n_bn_classes), torch.zeros(self.n_bn_classes)
        skews, kurts = torch.zeros(self.n_bn_classes), torch.zeros(self.n_bn_classes)

        layer_index = 0
        for name, block in self.bn_outputs.items():

            # Residual blocks are nested
            for sublayer in range(0, len(block), 1):
                dist = block[sublayer].flatten()
                std, mean = torch.std_mean(dist)
                skew = moment(dist, std, mean, deg=3)
                kurt = moment(dist, std, mean, deg=4)
                means[layer_index] = mean
                stds[layer_index] = std
                skews[layer_index] = skew
                kurts[layer_index] = kurt
                layer_index += 1
        return means, stds, skews, kurts

    def get_intermediate_layer_output(self, layer, inputs):
        layer_name = "layer"
        activation = {}

        def get_activation(name):
            def hook(self, input, output):
                print("calling hook")
                activation[name] = output.detach()

            return hook

        layer.register_forward_hook(
            get_activation(layer_name)
        )
        _ = self(inputs)
        return activation[layer_name]

def train_small_model():
    DATA_DIR = "../data"
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Pad(2)
    ])
    BATCH_SIZE = 512
    train_kwargs = {'batch_size': BATCH_SIZE}
    test_kwargs = {'batch_size': BATCH_SIZE}
    dataset1 = datasets.MNIST(DATA_DIR, train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST(DATA_DIR, train=False,
                       transform=transform)

    train_dl = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    val_dl = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    max_lr = 0.005
    weight_decay = 1e-5
    model = SmallModel()
    optimizer = torch.optim.Adam(model.parameters(),
                             max_lr,
                             weight_decay = weight_decay)
    EPOCHS = 13

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")

        train_loss = 0
        bn_means, bn_stds, bn_kurts = 0,0,0
        N = 0

        model.train()

        for i, (img, label) in enumerate(train_dl):
            logit = model(img)

            # Model loss
            loss = F.cross_entropy(logit,label)

            # Loss modifications
            bn_mean, bn_std, bn_kurt = model.get_bn_loss_metrics()
            loss += (bn_mean + bn_std + bn_kurt)

            loss.backward()

            # Save stuff
            train_loss += loss.item()
            bn_means += bn_mean.item()
            bn_stds += bn_std.item()
            bn_kurts += bn_kurt.item()
            N += 1

            optimizer.step()
            optimizer.zero_grad()

    print("Saving model as %s" % "small_model.pt")
    torch.save(model.state_dict(), "small_model.pt")

if __name__ == "__main__":
    train_small_model()
