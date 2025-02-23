import torch

def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7, abc=(3.4445, -4.7750,  2.0315)):
    assert len(G.shape) == 2
    a, b, c = abc
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

def orthogonalize(G):
    return zeroth_power_via_newtonschulz5(G, steps=10, eps=1e-8, abc=(3, -3.2, 1.2))

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                weight_decay=0.0, backend='newtonschulz5', backend_steps=5):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            backend=backend,
            backend_steps=backend_steps
        )
        super().__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeroth_power_via_newtonschulz5

            for i, p in enumerate(group['params']):
                g = p.grad
                #assert g is not None
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_backend(g, steps=group['backend_steps'])
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - lr * group['weight_decay'])
                #p.data.add_(g, alpha=-lr * max(1, (g.shape[-2] / g.shape[-1]))**0.5) #XXX
                p.data.add_(g, alpha=-lr)