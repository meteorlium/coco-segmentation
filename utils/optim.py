from torch import optim


def set_optim(net, method, lr):
    if method == "Adam":
        return optim.Adam(net.parameters(), lr=lr)
    elif method == "RMSprop":
        return optim.RMSprop(net.parameters(), lr=lr)
    else:
        return optim.SGD(net.parameters(), lr=lr, momentum=0.9)
