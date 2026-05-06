# 03 - Policy Architectures (MLP / RNN / MoE)

Relevant files:
- `networks/policy/actor_critic.py`
- `networks/policy/actor_critic_rnn.py`
- `networks/policy/actor_critic_moe.py`
- `networks/encoders/contact_encoder.py`
- `networks/encoders/lidar_encoder.py`

## 1) MLP baseline (`ActorCritic`)

- Standard actor/critic MLP.
- No recurrent state, no MoE.
- Used as a simple benchmark baseline.

## 2) RNN baseline (`ActorCriticRNN`)

- Adds an LSTM core.
- Uses token sequence `[ground_tokens | forward_tokens | proprio_token]`.
- Still does not include full MoE privileged fusion.

## 3) Main policy (`ActorCriticMoE`)

### Input split
- `p_t` (proprioception)
- `c_t` (commands)
- `e_t` (explicit privileged)
- `i_t` (implicit privileged/contact)
- LiDAR ground and forward point streams

### Implicit contact encoder
- `i_t -> ContactEncoder`
- Current shape example: `21 -> 32 -> 16 -> 4`
- Output latent: `z_t` (4D by default)

### Privileged latent assembly
- `l_t = [p_t, e_t, z_t, c_t]`
- `l_t` is projected into a proprio token.

### LiDAR context stream
- LiDAR encoder returns per-ray tokens.
- Sequence into LSTM:
  `[ground_tokens | forward_tokens | proprio_token]`

### MoE heads
- Gate network: `Linear -> ELU -> Linear -> softmax`.
- Multiple actor experts and critic experts.
- Final outputs are gated weighted sums.

## 4) Action noise and stability

- `log_std` is learnable.
- `clip_min_std` (currently `0.05`) prevents too-small std.
- `init_noise_std > 0` is validated.

## 5) Current dimensional notes

- `p_t = 42`
- `c_t = 6`
- `e_t` and `i_t` are inferred from runtime critic layout in `train_moe.py`.
- `ContactEncoder.contact_dim` is initialized from inferred `num_it_obs`.
