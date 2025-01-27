import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment
from gymnax.environments import spaces
from flax import struct
import chex
from jax import lax
from typing import Optional
import jax

jax.config.update("jax_enable_x64", True)  # TODO unsure if need or not but will check results


"""
E - Episodes
B - Batch
L - Sequence Length
A - Num Actions
S - Num States
O - Obs Dimension
K - idk what k is, its half states?
"""


@struct.dataclass
class EnvState(environment.EnvState):
    u: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):  # TODO sort this out at some point to match gymnax style
    S_DIM: int = 8
    A_DIM: int = 4
    A_MAX: int = 1
    L: int = 22
    dt: float = 0.05
    x: np.ndarray = np.loadtxt('project_name/envs/x.dat')  # select space discretization of the target solution
    U_bf: np.ndarray = np.loadtxt('project_name/envs/u2.dat')  # select u1, u2 or u3 as target solution


class KS_JAX(environment.Environment[EnvState, EnvParams]):
    def __init__(self):
        self.params = EnvParams()
        N = self.params.x.size
        self.dt = self.params.dt
        self.x_S = jnp.arange(N) * self.params.L / N
        k_K = N * jnp.fft.fftfreq(N)[0:N // 2 + 1] * 2 * jnp.pi / self.params.L
        self.ik_K = 1j * k_K                   # spectral derivative operator
        self.lin_K = k_K ** 2 - k_K ** 4       # Fourier multipliers for linear term
        self.a_dim = self.params.A_DIM
        self.s_dim = self.params.S_DIM
        sig = 0.4
        x_zero_A = self.x_S[-1] / self.params.A_DIM * jnp.arange(0, self.params.A_DIM)
        gaus = 1 / (jnp.sqrt(2 * jnp.pi) * sig) * jnp.exp(-0.5 * ((self.x_S - self.x_S[self.x_S.size // 2]) / sig) ** 2)

        def process_single(gaus, x_zero, x_S_center, dx):
            shift = jnp.floor(x_zero - x_S_center) / dx
            col = jnp.roll(gaus, shift.astype(int))
            col = col / jnp.max(col)
            return jnp.roll(col, 5)

        self.B_SA = jax.vmap(process_single, in_axes=(None, 0, None, None))(gaus,
                                                                       x_zero_A,
                                                                       self.x_S[self.x_S.size // 2],
                                                                       self.x_S[1] - self.x_S[0],
                                                                      ).T

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    def nlterm(self, u, f):
        # compute tendency from nonlinear term. advection + forcing
        ur = jnp.fft.irfft(u, axis=-1)
        return -0.5 * self.ik_K * jnp.fft.rfft(ur ** 2, axis=-1) + f

    # def advance(self, u0_S, action):
    def step_env(self, key: chex.PRNGKey, state: EnvState, action: jnp.ndarray, params: EnvParams):
        # forcing shape
        dum_SA = self.B_SA * action.T  # TODO check this
        f0_S = jnp.sum(dum_SA, axis=-1)

        # semi-implicit third-order runge kutta update.
        # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
        u_K = jnp.fft.rfft(state.u, axis=-1)
        f_K = jnp.fft.rfft(f0_S, axis=-1)
        u_save_K = u_K.copy()  # TODO is this required?

        def _runge_kutta_update(runner, unused):
            u_K, ind = runner
            dt = self.dt / (3 - ind)
            u_K = u_save_K + dt * self.nlterm(u_K, f_K)
            u_K = (u_K + 0.5 * self.lin_K * dt * u_save_K) / (1. - 0.5 * self.lin_K * dt)

            ind += 1

            return (u_K, ind), None

        final_runner_state = jax.lax.scan(_runge_kutta_update, (u_K, 0), None, 3)
        u_S = jnp.fft.irfft(final_runner_state[0][0], axis=-1)

        reward = -jnp.linalg.norm(u_S - self.params.U_bf)

        state = EnvState(u=u_S,
                         time=state.time + 1)

        done = self.is_terminal(state, params)

        return lax.stop_gradient(self.get_obs(state)), lax.stop_gradient(state), reward, done, {"empty": None}

    def reset_env(self, key: chex.PRNGKey, params: EnvParams):
        u_S = np.loadtxt('project_name/envs/u3.dat')
        state = EnvState(u=u_S,
                         time=0)  # TODO is this okay?
        return self.get_obs(state), state

    def get_obs(self, state_S: EnvState, params=None, key=None):  # TODO this in case of partial observability
        return jnp.float32(state_S.u[5::self.x_S.shape[0] // self.s_dim])

    def is_terminal(self, state: EnvState, params: EnvParams):
        return False

    @property
    def name(self) -> str:
        """Environment name."""
        return "KS-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.params.A_DIM

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-self.params.A_MAX, self.params.A_MAX, self.params.A_DIM, dtype=jnp.float64)  # TODO 64 or 32 precision?

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = 10  # TODO unsure of actual size should check
        return spaces.Box(-high, high, self.params.S_DIM, dtype=jnp.float32)

    # def state_space(self, params: EnvParams) -> spaces.Dict:
    #     """State space of the environment."""
    #     high = jnp.array(
    #         [
    #             params.x_threshold * 2,
    #             jnp.finfo(jnp.float32).max,
    #             params.theta_threshold_radians * 2,
    #             jnp.finfo(jnp.float32).max,
    #         ]
    #     )
    #     return spaces.Dict(
    #         {
    #             "x": spaces.Box(-high[0], high[0], (), jnp.float32),
    #             "x_dot": spaces.Box(-high[1], high[1], (), jnp.float32),
    #             "theta": spaces.Box(-high[2], high[2], (), jnp.float32),
    #             "theta_dot": spaces.Box(-high[3], high[3], (), jnp.float32),
    #             "time": spaces.Discrete(params.max_steps_in_episode),
    #         }
    #     )
                
if __name__ == "__main__":
    U_bf = np.loadtxt('u2.dat')  # select u1, u2 or u3 as target solution
    x = np.loadtxt('x.dat')  # select space discretization of the target solution

    MAX_EPISODES = 5000  # total epoch iterations
    MAX_STEPS = 2000  # total steps in one epoch
    MAX_TOTAL_REWARD = -16.5  # critic reward to Failure
    S_DIM = 8  # number of equispaced sensors
    A_DIM = 4  # number of equispaced actuators
    A_MAX = 1  # maximum amplitude for the actuation [-A_MAX, A_MAX]

    ks = KS_JAX(L=22, N=x.size, a_dim=A_DIM, s_dim=S_DIM)  # Kuramoto-Sivashinsky class initialization

    ini = 0
    with jax.disable_jit(disable=False):
        for _ep in range(ini, MAX_EPISODES):
            # TODO obs and state are the wrong way around here
            # new_observation = np.loadtxt('u3.dat')  # load initial condition
            obs_O, state_S = ks.reset_env(None, None)
            for r in range(MAX_STEPS):
                # state = jnp.float32(new_observation[5::x.shape[0] // S_DIM])
                # state = new_state
                action_A = jnp.zeros((A_DIM,))
                obs_O, state_S = ks.advance(state_S, action_A)
                print(obs_O)
