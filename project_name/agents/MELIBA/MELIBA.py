import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax.training.train_state import TrainState
import optax
from functools import partial
import sys
import flax.linen as nn
from typing import Tuple, NamedTuple, Mapping, Any
import flashbax as fbx
import distrax
from project_name.utils import remove_element
from project_name.agents import AgentBase
from ..MELIBA import get_MELIBA_config, HierarchicalSequentialVAE, EncoderRNN, DecoderRNN, Encoder, Decoder, PPOAgent


class MemoryStateMELIBA(NamedTuple):
    """State consists of network extras (to be batched)"""
    ppo_hstate: jnp.ndarray
    encoder_hstate: jnp.ndarray
    decoder_hstate: jnp.ndarray
    vae_buffer_state: jnp.ndarray
    extras: Mapping[str, jnp.ndarray]


class TrainStateMELIBA(NamedTuple):
    vae_state: TrainState
    ppo_state: TrainState


class TransitionMELIBA(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray


class TransitionNoMemState(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    latent: jnp.ndarray


class MELIBAAgent(AgentBase):
    def __init__(self, env,
                 env_params,
                 key: chex.PRNGKey,
                 config):
        self.config = config
        self.env = env
        self.env_params = env_params
        self.agent_config = get_MELIBA_config()

        init_x = (
            jnp.zeros((1, config.NUM_ENVS, env.observation_space(env_params).n)),  # states
            jnp.zeros((1, config.NUM_ENVS, config.NUM_AGENTS)),
            # env.action_space(env_params).n)),     # actions
            jnp.zeros((1, config.NUM_ENVS, 1)),  # rewards
            jnp.zeros((1, config.NUM_ENVS)))  # dones

        key, _key = jrandom.split(key)
        self.encoder = Encoder(config)
        self.decoder = Decoder(env.action_space(env_params).n, config)
        self.hsvae = HierarchicalSequentialVAE(env.action_space(env_params).n, config)

        self.init_encoder_hstate = EncoderRNN.initialize_carry(config.NUM_ENVS,
                                                               self.agent_config.GRU_HIDDEN_DIM)
        self.init_decoder_hstate = DecoderRNN.initialize_carry(config.NUM_ENVS, self.agent_config.GRU_HIDDEN_DIM)

        self.hsvae_params = self.hsvae.init(_key, self.init_encoder_hstate, self.init_decoder_hstate, init_x, key)

        self.ppo = PPOAgent(env, env_params, key, config, self.agent_config)

        self.vae_buffer = fbx.make_trajectory_buffer(max_length_time_axis=config.NUM_INNER_STEPS * 100,  # TODO is this okay?
                                                           min_length_time_axis=0,  # TODO again is this okay?
                                                           sample_batch_size=config.NUM_ENVS,
                                                           add_batch_size=config.NUM_ENVS,  # TODO is this right?
                                                           sample_sequence_length=config.NUM_INNER_STEPS+1,
                                                           period=1,  # TODO again is this okay?
                                                     )

        self.vae_buffer = self.vae_buffer.replace(init=jax.jit(self.vae_buffer.init),
                                                  add=jax.jit(self.vae_buffer.add, donate_argnums=0),
                                                  sample=jax.jit(self.vae_buffer.sample),
                                                  can_sample=jax.jit(self.vae_buffer.can_sample),
                                                  )

    def create_train_state(self):  # TODO seperate train state for vae and ppo?
        ppo_state = self.ppo.create_train_state()
        return (TrainStateMELIBA(vae_state=TrainState.create(apply_fn=self.hsvae.apply,
                                                             params=self.hsvae_params,
                                                             tx=optax.adam(self.agent_config.LR)),
                                 ppo_state=ppo_state[0]),
                MemoryStateMELIBA(ppo_hstate=ppo_state[1],
                                  encoder_hstate=self.init_encoder_hstate,
                                  decoder_hstate=self.init_decoder_hstate,
                                  vae_buffer_state=self.vae_buffer.init(
                                      TransitionMELIBA(done=jnp.zeros((), dtype=bool),
                                                       action=jnp.zeros((self.config.NUM_AGENTS),
                                                                        dtype=jnp.int32),
                                                       reward=jnp.zeros(()),
                                                       obs=jnp.zeros((self.env.observation_space(self.env_params).n),
                                                                     dtype=jnp.int8),
                                                       # TODO is it always an int for the obs?
                                                       )),
                                  extras={
                                      "values": jnp.zeros((self.config.NUM_ENVS, 1)),
                                      "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
                                      "latent_sample": jnp.zeros((1, self.config.NUM_ENVS, self.agent_config.LATENT_DIM)),
                                      "latent_mean": jnp.zeros((1, self.config.NUM_ENVS, self.agent_config.LATENT_DIM)),
                                      "latent_logvar": jnp.zeros((1, self.config.NUM_ENVS, self.agent_config.LATENT_DIM)),
                                  }, ),
                )

    @partial(jax.jit, static_argnums=(0,))
    def reset_memory(self, mem_state):
        # TODO get batch of data from vae storage

        # TODO run encoder over it but needs hidden state to be reset here maybe?

        # TODO add the output latent to the mem_state

        mem_state = mem_state._replace(extras={
            "values": jnp.zeros((self.config.NUM_ENVS, 1)),
            "log_probs": jnp.zeros((self.config.NUM_ENVS, 1)),
            "latent_sample": jnp.zeros((1, self.config.NUM_ENVS, self.agent_config.LATENT_DIM)),
            "latent_mean": jnp.zeros((1, self.config.NUM_ENVS, self.agent_config.LATENT_DIM)),
            "latent_logvar": jnp.zeros((1, self.config.NUM_ENVS, self.agent_config.LATENT_DIM)),
            # TODO should these reset to 0 or not idk?
        },
            ppo_hstate=self.ppo.create_train_state()[1],
            encoder_hstate=jnp.zeros((self.config.NUM_ENVS, self.agent_config.GRU_HIDDEN_DIM)),
            decoder_hstate=jnp.zeros((self.config.NUM_ENVS, self.agent_config.GRU_HIDDEN_DIM)),
        )
        return mem_state

    @partial(jax.jit, static_argnums=(0,))
    def act(self, train_state: Any, mem_state: Any, ac_in: chex.Array, key: chex.PRNGKey):
        # obs, dones = ac_in
        # new_obs = jnp.concatenate((obs, belief), axis=-1)  # TODO should we embed the input or not? this would make it trickier if so
        # ac_in = new_obs, dones

        latent_sample = mem_state.extras["latent_sample"]
        latent_mean = mem_state.extras["latent_mean"]
        latent_logvar = mem_state.extras["latent_logvar"]

        # def _get_latent_for_policy(latent_sample=None, latent_mean=None, latent_logvar=None):
        #
        #     if (latent_sample is None) and (latent_mean is None) and (latent_logvar is None):
        #         return None
        #
        #     if args.add_nonlinearity_to_latent:
        #         latent_sample = F.relu(latent_sample)
        #         latent_mean = F.relu(latent_mean)
        #         latent_logvar = F.relu(latent_logvar)
        #
        #     if args.sample_embeddings:
        #         latent = latent_sample
        #     else:
        #         latent = torch.cat((latent_mean, latent_logvar), dim=-1)
        #
        #     if latent.shape[0] == 1:
        #         latent = latent.squeeze(0)
        #
        #     return latent

        # TODO get latent for policy
        # latent = _get_latent_for_policy(latent_sample=latent_sample, latent_mean=latent_mean,
        #                                latent_logvar=latent_logvar)
        latent = latent_sample

        # TODO add latent to something to feed into the policy

        # needs to act via ppo agent but also takes in posterior from encoder
        mem_state, action, log_prob, value, key = self.ppo.act(train_state.ppo_state, mem_state, ac_in, latent, key)

        # # TODO update encoding maybe
        # latent_sample, latent_mean, latent_logvar, encoder_hstate = self.hsvae.encoder()

        return mem_state, action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0,2))
    def update(self, runner_state: chex.Array, agent, traj_batch: chex.Array):
        train_state, mem_state, env_state, ac_in, key = runner_state

        # # print(traj_batch)
        # dones = traj_batch.done
        # actions = traj_batch.action
        # print(dones)
        # print(actions)
        # sys.exit()
        #
        # def body_fn(carry, x):
        #     trajectory, collecting, length = carry
        #     done = x[done_idx]
        #
        #     # If we are collecting and haven't hit 'done'
        #     collecting = collecting | (length == 0)  # Start collecting if length is 0
        #     length += jnp.where(collecting, 1, 0)  # Increase length if collecting
        #
        #     # Add to trajectory buffer if collecting
        #     trajectory = jnp.where(collecting, trajectory.at[length - 1].set(x), trajectory)
        #
        #     # If done flag is encountered, reset for the next trajectory
        #     collecting = jnp.where(done, False, collecting)
        #     length = jnp.where(done, 0, length)
        #
        #     return (trajectory, collecting, length), (trajectory, done)
        #
        # (final_trajectory, _, _), (trajectories, done_flags) = lax.scan(body_fn, initial_state, data)
        #
        # sys.exit()



        new_buffer_state = self.vae_buffer.add(mem_state.vae_buffer_state,
                                               TransitionMELIBA(done=jnp.swapaxes(traj_batch.done[:, agent, :], 0, 1),
                                                                action=jnp.swapaxes(jnp.swapaxes(traj_batch.action, 1, 2), 0, 1),
                                                                reward=jnp.swapaxes(traj_batch.reward[:, agent, :], 0, 1),
                                                                obs=jnp.swapaxes(traj_batch.obs[:, agent, :], 0, 1),
                                                                ))
        mem_state = mem_state._replace(vae_buffer_state=new_buffer_state)
        key, _key = jrandom.split(key)
        batch = self.vae_buffer.sample(mem_state.vae_buffer_state, _key)

        # update ppo loss
        ppo_runner_state = train_state.ppo_state, mem_state, env_state, ac_in, key
        # TODO check it works with latent tracking below as unsure if gets at the right step basically

        # TODO add the latent thing as for act as this can be either sample or combo of mean and stdddev etc
        ppo_traj_batch = TransitionNoMemState(traj_batch.global_done,
                                              traj_batch.done,
                                              traj_batch.action,
                                              traj_batch.value,
                                              traj_batch.reward,
                                              traj_batch.log_prob,
                                              traj_batch.obs,
                                              traj_batch.info,
                                              traj_batch.mem_state.extras["latent_sample"].squeeze(axis=1))
        ppo_train_state, mem_state, env_state, ac_in, key = self.ppo.update(ppo_runner_state, agent, ppo_traj_batch)
        train_state = train_state._replace(ppo_state=ppo_train_state)  # TODO check this works

        # then also update vae but don't pass through each other
        vae_train_state, total_loss = self._update_vae(train_state.vae_state, mem_state, agent, batch, key)
        train_state = train_state._replace(vae_state=vae_train_state)  # TODO check this works

        # TODO update encoding my dude encode_running_trajectory

        return train_state, mem_state, env_state, ac_in, key

    @partial(jax.jit, static_argnums=(0,3))
    def _update_vae(self, train_state, mem_state, agent, batch, key):

        def calc_kl_loss(mean, log_var):  # TODO unsure this is correct for now
            gaussian_dim = mean.shape[-1]
            means_inc_prior = jnp.concatenate((jnp.zeros((mean[0][jnp.newaxis, :].shape)), mean), axis=0)
            log_vars_inc_prior = jnp.concatenate((jnp.ones((log_var[0][jnp.newaxis, :].shape)), log_var), axis=0)
            # TODO is the above ones or zeros? varibad is ident matrix but in paper it says 1s?

            posterior_means = means_inc_prior[1:]
            prior_means = means_inc_prior[:-1]

            posterior_log_vars = log_vars_inc_prior[1:]
            prior_log_vars = log_vars_inc_prior[:-1]

            kl_divergences = 0.5 * (
                    jnp.sum(prior_log_vars, axis=-1) - jnp.sum(posterior_log_vars, axis=-1) - gaussian_dim +
                    jnp.sum(1 / jnp.exp(prior_log_vars) * jnp.exp(posterior_log_vars), axis=-1) +
                    jnp.sum(((prior_means - posterior_means) / jnp.exp(prior_log_vars) *
                             (prior_means - posterior_means)), axis=-1))

            return kl_divergences

        def compute_loss(params, hidden_encoder, hidden_decoder, batch, key):  # TODO do I need this hidden state idk?
            # TODO this kinda dodgy loss I think, as t may shift but H stays the same? maybe that is okay?

            obs = jnp.swapaxes(batch.experience.obs, 0 ,1)[:-1]
            action = jnp.swapaxes(batch.experience.action, 0 ,1)[:-1]
            naction = jnp.swapaxes(batch.experience.action, 0 ,1)[1:]
            reward = jnp.expand_dims(jnp.swapaxes(batch.experience.reward, 0 ,1), axis=-1)[:-1]
            done = jnp.swapaxes(batch.experience.done, 0 ,1)[:-1]

            naction_opp = remove_element(naction, agent)

            # pass through encoder to get the mean and stddev
            future_action_logits, batch_samples, batch_mu, batch_logvar, _, _ = train_state.apply_fn(
                params,
                hidden_encoder,
                hidden_decoder,
                (obs, action, reward, done),
                key,
                full_traj=True)
            # TODO can we replace this with just encoder, how do the gradients flow then though?


            # train by elbo loss which will be dependent on t, aka an elbo for each t, includes prior in kl calc
            total_action_loss = []
            # batch_mu shape is num_steps, num_envs, latent_dim
            n_elbos = batch_mu.shape[0]

            for elbo in range(0, n_elbos):  # TODO jaxify this, maybe a scan?
                # # do a loss between predicted actions and future actions
                true_nactions_window = jnp.squeeze(naction_opp[elbo:], axis=-1)

                # print(future_action_logits[elbo:])
                # print(true_nactions_window)
                # print(optax.softmax_cross_entropy_with_integer_labels(future_action_logits[elbo:], true_nactions_window))
                # print("NEW ONE")

                # cross entropy loss as using discrete actions atm, prod over elbo terms, average over batches
                action_loss = jnp.mean(jnp.sum(
                    optax.softmax_cross_entropy_with_integer_labels(future_action_logits[elbo:], true_nactions_window), axis=0))
                # TODO is the above prod the right thing to do?

                print(action_loss)

                total_action_loss.append(action_loss)

            # # second loss which should be more efficient
            # def generate_masked_matrix(tensor):
            #     shape = tensor.shape
            #     n = shape[0]
            #
            #     def mask_step(k, tensor):
            #         mask = jnp.arange(n) >= k
            #         mask = mask.reshape(-1, *([1] * (len(shape) - 1)))
            #         mask = jnp.broadcast_to(mask, shape)
            #         return jnp.where(mask, tensor, 0)  # if use sum = 0, if use prod = 1
            #
            #     masked_tensors = jax.vmap(mask_step, in_axes=(0, None))(jnp.arange(n), tensor)
            #     return masked_tensors
            #
            # # cross entropy loss as using discrete actions atm
            # action_loss = optax.softmax_cross_entropy_with_integer_labels(future_action_logits,
            #                                                               jnp.squeeze(naction_opp, axis=-1))
            # # prod over individual elbo terms, average over batches, sum over all elbos
            # total_action_loss_two = jnp.sum(jnp.mean(jnp.sum(generate_masked_matrix(action_loss), axis=1), axis=-1))

            # do the kl loss between distributions, "prod" over t for elbos, mean over batches
            kl_loss = jnp.mean(jnp.prod(calc_kl_loss(batch_mu, batch_logvar), axis=0))
            # TODO could be a prod or mean or sum?

            total_action_loss = jnp.sum(jnp.stack(total_action_loss))
            total_action_loss_two = total_action_loss
            print(total_action_loss)
            print("NEW ONE")

            loss = self.agent_config.KL_WEIGHT * kl_loss + total_action_loss

            pi = distrax.Categorical(logits=future_action_logits)
            key, _key = jrandom.split(key)
            future_actions = pi.sample(seed=_key)

            return loss, (kl_loss, total_action_loss, future_actions, total_action_loss_two)

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True, argnums=0)
        total_loss, grads = grad_fn(train_state.params,
                                    mem_state.encoder_hstate,
                                    mem_state.decoder_hstate,
                                    batch,
                                    key)
        train_state = train_state.apply_gradients(grads=grads)

        loss, (kl_loss, total_action_loss, future_actions, total_action_loss_two) = total_loss

        def callback(total_action_loss, total_action_loss_two):
            print(total_action_loss)
            print(total_action_loss_two)
            print("NEW ONE")

        # jax.experimental.io_callback(callback, None, total_action_loss, total_action_loss_two)

        return train_state, loss
