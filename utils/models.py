from torch import nn

class Simple(nn.Module):
    
    def __init__(self, x_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 2),
            nn.GELU(),
            nn.Linear(2, 5),
            nn.GELU()
        )
    
    def forward(self, x):
        
        return self.model(x)


def get_model(name: str, **kwargs):
    
    configured_models = [
        'simple'
    ]
    
    if name not in configured_models:
        raise NotImplementedError(f'{name} is not a configured model, `name` must be one of {configured_models}')
    
    if name == 'simple':
        
        return Simple(**kwargs)
    