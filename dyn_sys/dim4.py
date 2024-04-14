import torch

def coupled_brusselator(t, X):
    ''' https://www.sciencedirect.com/science/article/pii/S0960077923001418 '''
    x1, y1, x2, y2 = X
    a = 2.0
    b = 6.375300171526684
    lambda_1 = 1.2
    lambda_2 = 80.0
    dxdt = a - (1+b)*x1 + x1**2 * y1 + lambda_1*(x2 - x1)
    dydt = b*x1 - x1**2 * y1 + lambda_2*(y2 - y1)
    dx2dt = a - (1+b)*x2 + x2**2 * y2 + lambda_1*(x1 - x2)
    dy2dt = b*x2 - x2**2 * y2 + lambda_2*(y1 - y2)

    return torch.stack([dxdt, dydt, dx2dt, dy2dt])

# def hyperchaos(t, X):
#     '''An Equation for Hyperchaos by Rossler, Physics Letter, 1979'''
#     x, y, z, w = X

#     dxdt = -y-z
#     dydt = x + 0.25*y + w
#     dzdt = 3 + x*z
#     dwdt = -0.5*z + 0.05*w
#     return torch.stack([dxdt, dydt, dzdt, dwdt])

# '''def hyperchaos(t, X):
 
#     '''https://www.sciencedirect.com/science/article/pii/S096007790400431X'''
#     x1, x2, x3, x4 = X
#     a= 35
#     b= 10
#     c= 1
#     d= 17

#     dxdt = a*(x2-x1) + x2*x3*x4
#     dydt = b*(x1 + x2) - x1*x3*x4
#     dzdt = -c*x3 + x1*x2*x4
#     dwdt = -d*x4 + x1*x2*x3
#     return torch.stack([dxdt, dydt, dzdt, dwdt])'''

def hyperchaos(t, X):

    '''https://www.sciencedirect.com/science/article/pii/S0030402616314097?casa_token=2aJAtjBo1YYAAAAA:FAU1TeiM2GTI21yXy0sD8-ZJtMdAq62tBu9UpDSABd1b3mkpSV9ZvmRYe8lwJRv1WNg7HAIe'''
    ''' LS: [4.0387, 0.021024, -20.0044, -40.6202.] ''' 
    x, y, z, w = X
    a= 16
    b= 40
    c= 20
    d= 8

    dxdt = a*x +d*z - y*z
    dydt = x*z - b*y
    dzdt = c*(x-z) + x*y
    dwdt = c*(y-w) + x*z
    return torch.stack([dxdt, dydt, dzdt, dwdt])


def hyperchaos_hu(X):
    '''https://ieeexplore.ieee.org/document/8367792'''
    '''[3.39, 1.41. 0.75, 0.34, 0.19, 0, -8.3]'''
    A = torch.stack([torch.tensor([-0.5, -4.9, 5.1, 1., 1., 1., 1.]),
                     torch.tensor([4.9, -5.3, 0.1, 1., 1., 1., 1.]),
                     torch.tensor([-5.1, 0.1, 4.7, 1., -1., 1., 1.]),
                     torch.tensor([1., 2., -3., -0.05, -1., 1., 1.]),
                     torch.tensor([-1., 1., 1., 1., -0.5, -1., 1.]),
                     torch.tensor([1., -2., -3., -1., 1., -0.1, 1.]),
                     torch.tensor([-1., -1., 1., 1., -1., -1., -0.5])])
    b = torch.tensor([40 * torch.sin(4 * 40 * X[-1]), 0., 0., 0., 0., 0., 0.]).requires_grad_(True).float()
    res = torch.matmul(A, X.T.float()) + b
    # print(res.grad_fn)
    return res
