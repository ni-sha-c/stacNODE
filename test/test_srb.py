@cuda.jit(device=True)
def dynamics(x, s):
    sinx1, sinx2 = sin(x[0]), sin(2*x[1])
    ceilxpi = math.floor(x[0] / pi) * (2*pi)
    x[0] = (2 * x[0] + (s[0] + s[1] * sinx2 / 2) * sinx1 - ceilxpi) % (2*pi)
    x[1] = ((x[1] + (s[3] + s[2] * sinx1) * sinx2 + ceilxpi) * 3) % (2*pi)

@cuda.jit
def compute_srb(n_g, z0, s, srb):
    # Make a grid on [0,2*pi]
    o_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    n_t = 100
    beg = (o_id - 1) * n_g * n_g
    pi2 = 2*pi
    dz = pi2/n_g
    z = z0[o_id]
    for i in range(n_t):
        dynamics(z,s)
        x, y = z
        pos_x, pos_y = int(x/dz), int(y/dz)
        pos = beg + pos_x + pos_y * n_g
        srb[pos] += 1/n_t



def plot_srb(s, ng):
    pi2, ng2 = 2*pi, ng*ng
    dz = pi2/ng
    z = [((i + 0.5)*dz) for i in range(ng)]
    zx, zy = meshgrid(z, z)


    nth, nbl = 256, 256
    nt = nth*nbl
    srb = cuda.to_device(zeros(ng2*nt))
    z0 = cuda.to_device(pi2*random.rand(nt,2))

    s = cuda.to_device(s)
    nrep = 10000
    for i in range(nrep):
        print("Averaged ", i*nt*100, " samples")
        compute_srb[nth, nbl](ng, z0, s, srb)
    srb = srb.copy_to_host()
    srb_g = array([sum(srb[i::ng2])/nt/nrep for i in range(ng2)]).reshape(ng, ng)
    srb_g = srb_g/dz/dz

    fig, ax = subplots()
    c = ax.contourf(zx,zy,srb_g)
    cbar = colorbar(c)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)

