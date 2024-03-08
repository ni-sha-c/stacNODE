def lorenz(t, u, rho=28.0):
    """ Lorenz chaotic differential equation: du/dt = f(t, u)
    t: time T to evaluate system
    u: state vector [x, y, z] 
    return: new state vector in shape of [3]"""
    sigma = 10.0
    #rho = 28.0
    beta = 8/3
    du = torch.stack([
            sigma * (u[1] - u[0]),
            u[0] * (rho - u[2]) - u[1],
            (u[0] * u[1]) - (beta * u[2])
        ])
    return du


class ODE_Lorenz(nn.Module):
    '''Define Neural Network that approximates differential equation system of Chaotic Lorenz'''
    def __init__(self, y_dim=3, n_hidden=32*9):
        super(ODE_Lorenz, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32 * 9),
            nn.GELU(),
            nn.Linear(32 * 9, 64 * 9),
            nn.GELU(),
            nn.Linear(64 * 9, 3)
        )
        # self.t = torch.linspace(0, 0.01, 2)
    def forward(self, t, y):
        res = self.net(y)
        return res

JAC_path = 'model.pt'
neural_vf = ODEFunc(

