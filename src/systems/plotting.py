from typing import Callable, Iterable, Union
import matplotlib.pyplot as plt
import numpy as np
import control

from .base import SystemEnv



def plot_env_response(
    env: SystemEnv, x0: np.ndarray, control_law: Union[Callable, np.ndarray], ax=None,
    state_idx=None, action_idx=None,
    state_names=None, action_names=None,
    clip_actions: bool=True
):
    if ax is not None:
        plt.sca(ax)
    if state_idx is not None:
        assert len(state_names)==len(state_idx)
    if action_idx is not None:
        assert len(action_names)==len(action_idx)
    x, u, r = [], [], []
    env.reset()
    if x0 is not None:
        env.state = x0
    state = env.state
    done = False
    if isinstance(control_law, np.ndarray):
        policy = lambda x: -control_law @ x
    else:
        policy = lambda x: control_law.predict(x, deterministic=True)[0]
    while not done:
        action = policy(state)
        if clip_actions:
            action = np.clip(action, a_min=env.action_space.low, a_max=env.action_space.high)
        x.append(state)
        state, reward, done, info = env.step(action)
        u.append(info.get('u', action))
        r.append(reward)
        if done: break
    x = np.asarray(x).T
    u = np.asarray(u).T
    R = sum(r)
    t = np.arange(x.shape[1]) * env.dt
    state_idx = range(len(x)) if state_idx is None else state_idx
    for i, xi in enumerate(state_idx):
        label = 'x%d' % xi if state_names is None else state_names[i]
        plt.plot(t, x[xi], label=label, lw=1.5)
    plt.xlabel('time')
    plt.ylabel('state')
    lines1 = plt.gca().lines
    title = plt.gca().get_title()
    title += '(Reward: %.2f)' % R
    plt.title(title)
    plt.twinx()
    action_idx = range(len(u)) if action_idx is None else action_idx
    for i, ui in enumerate(action_idx):
        label = 'u%d' % ui if action_names is None else action_names[i]
        plt.plot(t, u[ui], label=label, ls=':')
    lines2 = plt.gca().lines
    plt.ylabel('action')
    plt.legend(handles=lines1+lines2)



def plot_sys_response(
    sys, x0, t_final=10, k=None, q=None, r=None, ax=None
):
    if ax is not None:
        plt.sca(ax)
    if isinstance(sys, control.NonlinearIOSystem):
            resp = control.input_output_response(
                sys, np.linspace(0, t_final, 100), U=0, x0=x0)
    else:
        resp = control.initial_response(
            sys, np.linspace(0, t_final, 100), x0)
    t, y, x = resp.time, resp.outputs, resp.states
    for i, xi in enumerate(x):
        plt.plot(t, xi, label='x%d' % i)
    plt.xlabel('time')
    plt.ylabel('state')
    lines1, lines2 = plt.gca().lines, []
    if k is not None:
        plt.twinx()
        u = -k @ x
        for i, ui in enumerate(u):
            plt.plot(t, ui, label='u%d' % i, ls=':')
        lines2 = plt.gca().lines
        plt.ylabel('action')
    plt.legend(handles=lines1+lines2)
    cost = None
    if q is not None:
        cost = (x.T @ q @ x).sum()
    if r is not None:
        if cost is None:
            cost = (u.T @ r @ u).sum()
        else:
            cost += (u.T @ r @ u).sum()
    if cost is not None: plt.title('Cost: %.2f' % cost)



def multiple_response_plots(
    fns: Iterable[Callable], fig=None, figsize=(6,8), max_axis_ratio: float=10,
):
    """
    Plot multiple functions in subplots, aligning their limits.

    Parameters
    ----------
    fns : Iterable[Callable]
        A list of functions, or an alternating sequence of title and plotting function,
        [fn, fn, fn] or ['title', fn, 'title', fn,...]
    fig : matplotlib.Figure, optional
        The figure to plot in, by default None
    figsize : Tuple[float, float], optional
        The size of figure (x, y), by default (6,8)
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    size = fig.get_size_inches()
    n = sum([1 if callable(fn) else 0 for fn in fns])
    i = 1
    add_plot = True
    for fn in fns:
        if add_plot:
            ax = fig.add_subplot(n, 1, i)
            plt.sca(ax)
        plt.sca(ax)
        if isinstance(fn, str):
            add_plot = False
            ax.set_title(fn)
            continue
        else:
            fn()
            i += 1
            add_plot=True
    axes = fig.axes
    axl = axes[::2]
    axr = axes[1::2]
    for axs in (axl, axr):
        miny, maxy = np.inf, -np.inf
        for ax in axs:
            y0, y1 = ax.get_ylim()
            miny = y0 if y0 < miny else miny
            maxy = y1 if y1 > maxy else maxy
        for ax in axs:
            ax.set_ylim(miny, maxy)
    plt.tight_layout()