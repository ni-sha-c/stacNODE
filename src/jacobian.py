def Jacobian_Matrix(input, sigma, r, b):
    ''' Jacobian Matrix of Lorenz '''

    x, y, z = input
    return torch.stack([torch.tensor([-sigma, sigma, 0]), torch.tensor([r - z, -1, -x]), torch.tensor([y, x, -b])])



def Jacobian_Brusselator(dyn_sys_f, x):
    ''' Jacobian Matrix of Coupled Brusselator '''
    a = 2.0
    b = 6.375300171526684
    lambda_1 = 1.2
    lambda_2 = 80.0
    x1, y1, x2, y2 = x

    matrix = torch.stack([
    torch.tensor([-(b+1) + 2*x1*y1 -lambda_1, x1**2, lambda_1, 0]), 
    torch.tensor([b-2*x1*y1, - x1**2 - lambda_2, 0, lambda_2]), 
    torch.tensor([lambda_1, 0, -(b+1) + 2*x2*y2 -lambda_1, x2**2]), 
    torch.tensor([0, lambda_2, b - 2*x2*y2, -x2**2 - lambda_2])])

    return matrix

def Jacobian_Henon(X):
    x, y = X
    a=1.4
    b=0.3

    return torch.stack([torch.tensor([- 2*a*x, 1]), torch.tensor([b, 0])])

def Jacobian_Hyperchaos(X):
    a= 35
    b= 10
    c= 1
    d= 17

    x1, x2, x3, x4 = X

    # return torch.stack([torch.tensor([-a, a, 0, 0]), torch.tensor([b, b, 0, 0]), torch.tensor([0, 0, -c, c]), torch.tensor([0, 0, 0, -d])])
    return torch.stack([torch.tensor([-a, a+x3*x4, x2*x4, x2*x3]), torch.tensor([b-x3*x4, b, -x1*x4, -x1*x3]), torch.tensor([x2*x4, x1*x4, -c, x1*x2]), torch.tensor([x2*x3, x1*x3, x1*x2, -d])])