import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, params):
        super(DNN, self).__init__()

        if params["activation"] == "relu":
            act_fn = nn.ReLU
        elif params["activation"] == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {params['activation']}")

        layers = []
        in_features = params["input_size"]
        for h in params["hidden_layers"]:
            layers.append(nn.Linear(in_features, h))
            layers.append(act_fn())
            if params["dropout"] > 0.0:
                layers.append(nn.Dropout(params["dropout"]))
            in_features = h
        layers.append(nn.Linear(in_features, params["output_size"]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
