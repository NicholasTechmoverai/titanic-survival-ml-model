import torch.nn as nn

SEX_MAP = {
    "female": 0,
    "male": 1
}

EMBARKED_MAP = {
    "S": 0,
    "C": 1,
    "Q": 2
}


#the neural network (the brain)
class TitanicModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
