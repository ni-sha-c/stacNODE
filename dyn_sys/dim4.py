import torch


def hyperchaos(t, X):

    '''Cite: https://www.sciencedirect.com/science/article/pii/S0030402616314097?casa_token=2aJAtjBo1YYAAAAA:FAU1TeiM2GTI21yXy0sD8-ZJtMdAq62tBu9UpDSABd1b3mkpSV9ZvmRYe8lwJRv1WNg7HAIe'''
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

