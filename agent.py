
from datetime import datetime
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from network import DeepQNetwork
from memory import ReplayBuffer


class DQNAgent(object):
    def __init__(self,
            env_name,  # Do define this, please.
            input_dims,
            n_actions,
            gamma=0.99,             # Defaults are paper values.
            lr=0.00025,
            mem_size=1000000,
            batch_size=32,
            epsilon=1,
            epsilon_min=0.1,
            epsilon_steps=1000000,
            replace=10000,

            optimizer='Adam',
            double=True,
            versioning=False,
        ):

        self.double = double

        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_steps = epsilon_steps
        self.epsilon_dec = (self.epsilon - self.epsilon_min) / epsilon_steps

        self.replace_target_cnt = replace
        self.action_space = [i for i in range(n_actions)]
        self.step_counter = 0

        self.memory = ReplayBuffer(mem_size)

        # Training network
        self.q_train = DeepQNetwork(
            input_dims,
            n_actions,
        )

        if 'RMSProp' in optimizer:
            self.optimizer = optim.RMSprop(self.q_train.parameters(),
                lr=0.00025,
                momentum=0.95,  # This breaks things
                alpha=0.95,
                eps=0.01  # This breaks things
            )
            # https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
            # According to this, there are different implementations of RMSProp, and using
            #   momentum with this one really makes the model bite the dust.
            # Turns out pytorch and deepmind used same version, so its parameters must be wrong...
            #
            # alpha - "smoothing constant", must be the weight in the moving weighted average.
            #   This would correspond to the "squared gradient (denominator) momentum described
            #   in the Deepmind paper.
            # eps - "term added to denominator to improve numerical stability".
            #   This corresponds to "min squared gradient" of Deepmind paper: "Constant added to
            #   the squared gradient in the denominator of the RMSProp update."
            # No weight decay because moving target.

            RMSProp_properties = []
            for prop in ['lr', 'momentum', 'alpha', 'eps']:
                RMSProp_properties.append(f"{prop}={self.optimizer.param_groups[0][prop]}")
            optimizer_properties = f"RMSProp({', '.join(RMSProp_properties)})"

        else:
            self.optimizer = optim.Adam(self.q_train.parameters(), lr=0.0000625)
            optimizer_properties = f"Adam(lr={self.optimizer.param_groups[0]['lr']})"
        
        self.loss = nn.SmoothL1Loss()  # Error clipping.

        self.q_target = DeepQNetwork(
            input_dims,
            n_actions,
        )


        self.representation = '\n\t'.join([
            f"{env_name}_{'Double' if self.double else 'Normal'}",
            f"mem({self.memory.mem_size})_replace({self.replace_target_cnt})_batch-size({self.batch_size})",
            optimizer_properties,
            f"epsilon({self.epsilon}) eps-min({self.epsilon_min}) eps-steps({self.epsilon_steps})",
            f"gamma({self.gamma})",
            # timestamp,
        ])
        self.name = '_'.join([
            f"{env_name}_{'Double' if self.double else 'Normal'}",
            # f"mem({self.memory.mem_size})_replace({self.replace_target_cnt})",
            optimizer_properties,
            f"timestamp({datetime.now().strftime('%dd-%H-%M-%S')})",
        ])
        if versioning:
            from os import mkdir
            mkdir(f'models/{self.name}')
        self.versioning = versioning


        self.rng = np.random.default_rng()  # For choosing action


    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            with T.no_grad():
                observation = np.array(observation)
                state = T.tensor(observation).to(self.q_train.device).unsqueeze(dim=0) / 255.0

                q_vals = self.q_train.forward(state)
                return T.argmax(q_vals).item()
        else:
            return self.rng.choice(self.action_space)


    def store_memory(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_train.device)
        rewards = T.tensor(reward).to(self.q_train.device)
        dones = T.tensor(done).to(self.q_train.device)
        actions = T.tensor(action).to(self.q_train.device)
        states_ = T.tensor(new_state).to(self.q_train.device)

        states = states / 255.0
        states_ = states_ / 255.0
            # This should be trivially quick on the GPU.

        return states, actions, rewards, states_, dones


    def learn(self):
        # print(self.memory.mem_counter)
        if self.memory.mem_counter < self.batch_size:
            return

        if self.step_counter % 4 == 0:  # Do zero_grad and optimize on first step, just to make sure things are cleared up.
            self.optimizer.zero_grad()

        # Replace target network
        if self.step_counter % self.replace_target_cnt == 0:
            self.q_target.load_state_dict(self.q_train.state_dict())

        states, actions, rewards, states_, dones = self.sample_memory()

        # Get the q_vals of the actions we took, using our training model.
        indices = np.arange(self.batch_size)  # array[0 to batch_size-1]
        q_pred_all = self.q_train.forward(states)  # dim: batch_size by n_actions
        q_pred = q_pred_all[indices, actions]


        with T.no_grad():

            # Double Deep Q learning
            if self.double:
                # print("Double")
                # Instead of choosing the best actions using the target network, we calculate them
                # with our training network.
                # We then use our target network to get the actual Q-values of these actions.

                # Get the best action in each state, using our training model.
                q_eval_all = self.q_train.forward(states_)
                q_eval_best_actions = q_eval_all.argmax(dim=1)
                    # Index of maximal value for each row/memory-state.

                # Calculate the q_val of each state using our next_state model
                q_next_all = self.q_target.forward(states_)

                # Retrieve the best q_vals from this list, as chosen by our training model.
                q_next_best = q_next_all[indices, q_eval_best_actions]

            # Regular Deep Q learning
            else:
                # print("Regular")
                q_next_all = self.q_target.forward(states_)#.max(dim=1)[0]
                q_next_best = q_next_all.max(dim=1)[0]


            # Inside no_grad
            q_next_best[dones] = 0.0
            q_target = rewards + self.gamma * q_next_best

        # Outside no_grad
        loss = self.loss(q_target, q_pred).to(self.q_train.device)
        
        loss.backward()

        if self.step_counter % 4 == 0:
            self.optimizer.step()

        self.step_counter += 1

        self.epsilon = max(self.epsilon - self.epsilon_dec,  self.epsilon_min)


    def save_models(self, version, suffix=None):
        name = f"models/{self.name}/{version}{f'_{suffix}' if suffix else ''}"

        T.save(self.q_train.state_dict(), name)

    def load_models(self, name):
        self.q_train.load_state_dict(T.load(f'models/{name}'))
        self.q_target.load_state_dict(T.load(f'models/{name}'))
