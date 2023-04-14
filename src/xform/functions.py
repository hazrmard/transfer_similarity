from typing import Callable, Union, Tuple

import numpy as np
import torch
from torch import nn
from torch.autograd.functional import jacobian
from tqdm.autonotebook import trange



class InvertibleFunction:
    """Use as InvertibleFunction() @ and InvertibleFunction.invert() @"""
    def __init__(self, func: Callable, inverse: Callable):
        self.func, self.inverse = func, inverse
    def __call__(self, *args):
        return self.func(*args)
    def invert(self):
        return InvertibleFunction(self.inverse, self.func)



class MatMulFunction:

    def __init__(self, func: Callable[[np.ndarray], np.ndarray]):
        self.func = func


    def __matmul__(self, *operands):
        operand = operands[0]
        return self.func(operand)


class DynamicsModel(nn.Module):

    def __init__(self, n_states, n_actions, dt, states_hidden=None,
        separable: bool=False, normalized: bool=False, n_layers=1, device=None):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.states_hidden = (n_states * 2) if states_hidden is None else states_hidden
        self.separable = separable
        self.normalized = normalized
        self.n_layers = n_layers
        self.dt = dt
        self.device  = torch.device(device if device is not None else \
            ("cuda:0" if torch.cuda.is_available() else "cpu"))

        layers = [(nn.Linear(self.states_hidden, self.states_hidden), nn.Tanh()) for l in range(n_layers)]
        layers = [l for ls in layers for l in ls] # hidden layer + activation layers
        layers.append(nn.Linear(self.states_hidden, self.n_states)) # output layer
        if self.normalized:
            layers.append(nn.Tanh())

        if self.separable:
            layers = [nn.Linear(n_states, self.states_hidden), nn.Tanh()] + layers
            self.A = nn.Sequential(*layers)
            self.B = nn.Parameter(nn.init.xavier_uniform_(torch.empty((n_states, n_actions))))
            def dxdt(xu):
                x = xu[:, :self.n_states]
                u = xu[:, self.n_states:]
                return self.A(x) + (self.B @ u.T).T
        else:
            layers = [nn.Linear(n_states+n_actions, self.states_hidden), nn.Tanh()] + layers
            self.dP = nn.Sequential(*layers)
            def dxdt(xu):
                return self.dP(xu)
        self.dxdt = dxdt
        self.to(self.device)


    def forward(self, xu: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xu : torch.Tensor
            Concatenated tensor of shape (N measurements x [states + actions])

        Returns
        -------
        torch.Tensor
            Tensor of next state of shape (N measurements x states)
        """
        dx = self.dxdt(xu)
        x = xu[:, :self.n_states]
        return x + dx * self.dt


    def learn(self, xu: np.ndarray, x: np.ndarray, lr=1e-3, steps=100):
        xu, x = torch.from_numpy(xu.T).to(self.device), torch.from_numpy(x.T).to(self.device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss = nn.MSELoss()
        losses = []
        for i in trange(steps, leave=False):
            opt.zero_grad()
            x_ = self.forward(xu)
            l = loss(x, x_)
            l.backward()
            opt.step()
            losses.append(l.item())
        
        # def funcA(mat):
        #     if isinstance(mat, np.ndarray):
        #         mat = torch.from_numpy(mat).float()
        #     with torch.no_grad():
        #         return self.A(mat).detach().cpu().numpy()
        # A = MatMulFunction(funcA)

        # # A and B as matrix multiplication compatible objects, dxdt = A@x + B@u
        # return A, self.B.clone().detach().cpu().numpy(), np.asarray(losses)
        return np.asarray(losses)


    def ddxu(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        u = torch.as_tensor(u, dtype=torch.float32, device=self.device)
        xu = torch.cat((x,u), dim=-1)
        with torch.set_grad_enabled(True):
            xu = torch.atleast_2d(xu)
            xu.requires_grad_(True)
            # action, value, _ = self.forward(obs, deterministic=deterministic)
            # vgrad = dtensor_dx(value, obs)
            # agrad = dtensor_dx(action, obs)
            # https://discuss.pytorch.org/t/what-do-the-dimensions-of-the-output-of-torch-autograd-functional-jacobian-represent/144492
            ddxu = jacobian(self.dxdt, xu)
            ddxu = torch.einsum("bibj->bij", ddxu)
            ddxu = ddxu.cpu().numpy()
            ddx, ddu = ddxu[...,:self.n_states], ddxu[...,-self.n_actions:]
            return ddx, ddu
