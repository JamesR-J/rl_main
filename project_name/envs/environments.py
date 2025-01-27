import jax
import jax.random as jrandom
import jax.numpy as jnp
from functools import partial
import sys


def ipd_batched(bs, gamma_inner=0.96):  # TODO add checks between this and torch version
    dims = [5, 5]
    payout_mat_1 = jnp.array([[-1, -3], [0, -2]])
    payout_mat_2 = payout_mat_1.T
    payout_mat_1 = jnp.tile(payout_mat_1.reshape((1, 2, 2)), (bs, 1, 1))
    payout_mat_2 = jnp.tile(payout_mat_2.reshape((1, 2, 2)), (bs, 1, 1))

    def Ls(th):  # th is a list of two different tensors. First one is first agent? tnesor size is List[Tensor(bs, 5), Tensor(bs,5)].
        p_1_0 = jax.nn.sigmoid(th[0][:, 0:1])
        p_2_0 = jax.nn.sigmoid(th[1][:, 0:1])
        p = jnp.concatenate([p_1_0 * p_2_0, p_1_0 * (1 - p_2_0), (1 - p_1_0) * p_2_0, (1 - p_1_0) * (1 - p_2_0)],
                            axis=-1)
        p_1 = jnp.reshape(jax.nn.sigmoid(th[0][:, 1:5]), (bs, 4, 1))
        p_2 = jnp.reshape(
            jax.nn.sigmoid(jnp.concatenate([th[1][:, 1:2], th[1][:, 3:4], th[1][:, 2:3], th[1][:, 4:5]], axis=-1)),
            (bs, 4, 1))
        P = jnp.concatenate([p_1 * p_2, p_1 * (1 - p_2), (1 - p_1) * p_2, (1 - p_1) * (1 - p_2)], axis=-1)

        M = jnp.matmul(jnp.expand_dims(p, axis=1), jnp.linalg.inv(jnp.eye(4) - gamma_inner * P))
        L_1 = -jnp.matmul(M, jnp.reshape(payout_mat_1, (bs, 4, 1)))
        L_2 = -jnp.matmul(M, jnp.reshape(payout_mat_2, (bs, 4, 1)))

        return L_1.squeeze(-1), L_2.squeeze(-1), M

    return dims, Ls


# def imp_batched(bs, gamma_inner=0.96):
#     dims = [5, 5]
#     payout_mat_1 = torch.Tensor([[-1, 1], [1, -1]]).to(device)
#     payout_mat_2 = -payout_mat_1
#     payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)
#     payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)
#
#     def Ls(th):  # th is a list of two different tensors. First one is first agent? tnesor size is List[Tensor(bs, 5), Tensor(bs,5)].
#         p_1_0 = torch.sigmoid(th[0][:, 0:1])
#         p_2_0 = torch.sigmoid(th[1][:, 0:1])
#         p = torch.cat([p_1_0 * p_2_0, p_1_0 * (1 - p_2_0), (1 - p_1_0) * p_2_0, (1 - p_1_0) * (1 - p_2_0)], dim=-1)
#         p_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (bs, 4, 1))
#         p_2 = torch.reshape(torch.sigmoid(torch.cat([th[1][:, 1:2], th[1][:, 3:4], th[1][:, 2:3], th[1][:, 4:5]], dim=-1)), (bs, 4, 1))
#         P = torch.cat([p_1 * p_2, p_1 * (1 - p_2), (1 - p_1) * p_2, (1 - p_1) * (1 - p_2)], dim=-1)
#
#         M = torch.matmul(p.unsqueeze(1), torch.inverse(torch.eye(4).to(device) - gamma_inner * P))
#         L_1 = -torch.matmul(M, torch.reshape(payout_mat_1, (bs, 4, 1)))
#         L_2 = -torch.matmul(M, torch.reshape(payout_mat_2, (bs, 4, 1)))
#
#         return [L_1.squeeze(-1), L_2.squeeze(-1), M]
#
#     return dims, Ls
#
#

#
#
# def compute_best_response(outer_th_ba):
#     batch_size = 1
#     std = 0
#     num_steps = 1000
#     lr = 1
#
#     ipd_batched_env = ipd_batched(batch_size, gamma_inner=0.96)[1]
#     inner_th_ba = torch.nn.init.normal_(torch.empty((batch_size, 5), requires_grad=True), std=std).cuda()
#     for i in range(num_steps):
#         th_ba = [inner_th_ba, outer_th_ba.detach()]
#         l1, l2, M = ipd_batched_env(th_ba)
#         grad = get_gradient(l1.sum(), inner_th_ba)
#         with torch.no_grad():
#             inner_th_ba -= grad * lr
#     print(l1.mean() * (1 - 0.96))
#     return inner_th_ba
#
#
# def matching_pennies_batch(batch_size=128):
#     dims = [1, 1]
#     payout_mat_1 = torch.Tensor([[1, -1], [-1, 1]]).to(device)
#     payout_mat_2 = -payout_mat_1
#     payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(batch_size, 1, 1)
#     payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(batch_size, 1, 1)
#
#     def Ls(th):
#         p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
#         x, y = torch.cat([p_1, 1 - p_1], dim=-1), torch.cat([p_2, 1 - p_2], dim=-1)
#         L_1 = torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_1), y.unsqueeze(-1))
#         L_2 = torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_2), y.unsqueeze(-1))
#         return [L_1.squeeze(-1), L_2.squeeze(-1)]
#
#     return dims, Ls
#
#
# def chicken_game_batch(batch_size=128):
#     dims = [1, 1]
#     payout_mat_1 = torch.Tensor([[0, -1], [1, -100]]).to(device)
#     payout_mat_2 = torch.Tensor([[0, 1], [-1, -100]]).to(device)
#     payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(batch_size, 1, 1)
#     payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(batch_size, 1, 1)
#
#     def Ls(th):
#         p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
#         x, y = torch.cat([p_1, 1 - p_1], dim=-1), torch.cat([p_2, 1 - p_2], dim=-1)
#         L_1 = -torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_1), y.unsqueeze(-1))
#         L_2 = -torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_2), y.unsqueeze(-1))
#         return [L_1.squeeze(-1), L_2.squeeze(-1), None]
#
#     return dims, Ls


class MetaGames:
    def __init__(self, b, opponent="NL", game="IPD", mmapg_id=0):
        """
        Opponent can be:
        NL = Naive Learner (gradient updates through environment).
        LOLA = Gradient through NL.
        STATIC = Doesn't learn.
        COPYCAT = Copies what opponent played last step.
        """
        self.gamma_inner = 0.96
        self.b = b

        self.game = game
        if self.game == "IPD":
            d, self.game_batched = ipd_batched(b, gamma_inner=self.gamma_inner)
            self.std = 1
            self.lr = 1
        # elif self.game == "IMP":
        #     d, self.game_batched = imp_batched(b, gamma_inner=self.gamma_inner)
        #     self.std = 1
        #     self.lr = 1
        # elif self.game == "chicken":
        #     d, self.game_batched = chicken_game_batch(b)
        #     self.std = 1
        #     self.lr = 1
        # else:
        #     raise NotImplementedError
        self.d = d[0]

        self.opponent = opponent
        if self.opponent == "MAMAML":
            pass
            # f = f"data/mamaml_{self.game}_{mmapg_id}.th"
            # assert osp.exists(f), "Generate the MAMAML weights first"
            # self.init_th_ba = torch.load(f)
        else:
            self.init_th_ba = None

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, info=False):
        key, _key = jrandom.split(key)
        if self.init_th_ba is not None:
            print("TO SORT OUT")
            sys.exit()
            # self.inner_th_ba = self.init_th_ba.detach() * torch.ones((self.b, self.d), requires_grad=True)
        else:
            inner_th_ba = jrandom.normal(_key, shape=(self.b, self.d)) * self.std
        key, _key = jrandom.split(key)
        outer_th_ba = jrandom.normal(_key, shape=(self.b, self.d)) * self.std
        # inner_th_ba = jnp.array([[ 0.7754, -0.5341, -1.2947, -0.6062, -2.0508],
        # [ 0.0149, -1.2983,  0.3702, -1.2887,  0.4085],
        # [ 0.2923, -1.1933,  0.0612,  1.2291,  0.9198],
        # [ 0.9375,  0.6995,  0.1913, -1.0763,  0.1085]])
        # outer_th_ba = jnp.array([[-0.0297,  1.1426, -0.3348, -0.6799, -0.0866],
        # [-0.3019,  0.6254, -1.6504,  0.5789,  2.1183],
        # [-0.3985,  0.6669,  1.0110,  0.4311,  1.1836],
        # [-0.8658,  0.4625, -0.3750, -2.2171,  0.4537]])
        # print(inner_th_ba)
        # print(outer_th_ba)
        inner_th_ba, state, _, _, M = self.step(inner_th_ba, outer_th_ba)

        if info:
            return inner_th_ba, state, M
        else:
            return inner_th_ba, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, inner_th_ba, outer_th_ba):
        last_inner_th_ba = inner_th_ba
        th_ba = jnp.array((inner_th_ba, outer_th_ba))
        if self.opponent == "NL" or self.opponent == "MAMAML":
            def loss_function(vals):
                l1, l2, M = self.game_batched(vals)
                return jnp.sum(l1), (l1, l2, M)  # TODO do need stop gradients idk?
            (l1_loss, (l1, l2, M)), grads = jax.value_and_grad(loss_function, has_aux=True)(th_ba)
            grads = grads[0]
            inner_th_ba -= grads * self.lr
        # elif self.opponent == "LOLA":
        #     th_ba = [self.inner_th_ba, outer_th_ba.detach()]
        #     th_ba[1].requires_grad = True
        #     l1, l2, M = self.game_batched(th_ba)
        #     losses = [l1, l2]
        #     grad_L = [[get_gradient(losses[j].sum(), th_ba[i]) for j in range(2)] for i in range(2)]
        #     term = (grad_L[1][0] * grad_L[1][1]).sum()
        #     grad = grad_L[0][0] - self.lr * get_gradient(term, th_ba[0])
        #     with torch.no_grad():
        #         self.inner_th_ba -= grad * self.lr
        # elif self.opponent == "BR":
        #     num_steps = 1000
        #     inner_th_ba = torch.nn.init.normal_(torch.empty((self.b, 5), requires_grad=True), std=self.std).to(device)
        #     for i in range(num_steps):
        #         th_ba = [inner_th_ba, outer_th_ba.detach()]
        #         l1, l2, M = self.game_batched(th_ba)
        #         grad = get_gradient(l1.sum(), inner_th_ba)
        #         with torch.no_grad():
        #             inner_th_ba -= grad * self.lr
        #     with torch.no_grad():
        #         self.inner_th_ba = inner_th_ba
        #         th_ba = [self.inner_th_ba, outer_th_ba.detach()]
        #         l1, l2, M = self.game_batched(th_ba)
        else:
            raise NotImplementedError

        if self.game == "IPD" or self.game == "IMP":  # TODO also return the inner_th_bas stuff I guess
            return inner_th_ba, jax.nn.sigmoid(jnp.concatenate((outer_th_ba, last_inner_th_ba), axis=-1)), (
                    -l2 * (1 - self.gamma_inner)), (-l1 * (1 - self.gamma_inner)), M
        else:
            return inner_th_ba, jax.nn.sigmoid(
                jnp.concatenate((outer_th_ba, last_inner_th_ba), axis=-1)), -l2, -l1, M


if __name__ == "__main__":
    with jax.disable_jit():
        batch_size = 4  # 4096
        env = MetaGames(b=batch_size, opponent="NL", game="IPD", mmapg_id=0)
        max_episodes = 10
        num_steps = 100
        rew_means = []
        key = jrandom.PRNGKey(0)
        for i_episode in range(1, max_episodes + 1):
            key, _key = jrandom.split(key)  # TODO need this split or not?
            inner_stuff, state = env.reset(key)
            running_reward = jnp.zeros(batch_size)
            running_opp_reward = jnp.zeros(batch_size)

            last_reward = 0

            for t in range(num_steps):
                # Running policy_old:
                action = 0  # random action
                inner_stuff, state, reward, info, M = env.step(inner_stuff, action)

                running_reward += reward.squeeze(-1)
                running_opp_reward += info.squeeze(-1)
                last_reward = reward.squeeze(-1)

            print("=" * 100, flush=True)

            print(f"episode: {i_episode}", flush=True)

            print(f"loss: {-running_reward.mean() / num_steps}", flush=True)

            rew_means.append(
                {
                    "rew": (running_reward.mean() / num_steps).item(),
                    "opp_rew": (running_opp_reward.mean() / num_steps).item(),
                }
            )

            print(f"opponent loss: {-running_opp_reward.mean() / num_steps}", flush=True)
