from typing import Callable

import numpy as np
import torch
from torch import nn
from tqdm.autonotebook import trange



class InvertibleFunction:
    """Use as InvertibleFunction() @ and InvertibleFunction.invert() @"""
    def __init__(self, func: Callable, inverse: Callable):
        self.func, self.inverse = func, inverse
    def __call__(self, *args):
        return self.func(*args)
    def __matmul__(self, *args):
        return self(*args)
    def invert(self):
        return InvertibleFunction(self.inverse, self.func)



class MatMulFunction:

    def __init__(self, func):
        self.func = func
    def __matmul__(self, *args):
        return self.func(args[0])



class DynamicsModel(nn.Module):

    def __init__(self, n_states, n_actions, dt, states_hidden=None):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.states_hidden = (n_states * 2) if states_hidden is None else states_hidden
        self.dt = dt
        self.A = nn.Sequential(
            nn.Linear(n_states, self.states_hidden),
            nn.Tanh(),
            nn.Linear(self.states_hidden, self.states_hidden),
            nn.Tanh(),
            nn.Linear(self.states_hidden, n_states),
        )
        self.B = nn.Parameter(nn.init.xavier_uniform_(torch.empty((n_states, n_actions))))


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
        x = xu[:, :self.n_states]
        u = xu[:, self.n_states:]
        dx = self.A(x) + (self.B @ u.T).T
        return x + dx * self.dt


    def learn(self, xu, x, lr=1e-3, steps=100):
        xu, x = torch.from_numpy(xu.T), torch.from_numpy(x.T)
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
        
        def funcA(mat):
            if isinstance(mat, np.ndarray):
                mat = torch.from_numpy(mat).float()
            with torch.no_grad():
                return self.A(mat).detach().cpu().numpy()
        A = MatMulFunction(funcA)

        return A, self.B.clone().detach().cpu().numpy(), np.asarray(losses)
