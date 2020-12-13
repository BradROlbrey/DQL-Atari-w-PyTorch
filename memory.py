
import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.mem_size = max_size
        self.mem_counter = 0

        self.state_memory       = [[] for _ in range(self.mem_size)]
        self.new_state_memory   = [[] for _ in range(self.mem_size)]
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
            # terminal states will have a different target, treated differently.


    def store_transition(self, state, action, reward, state_, done):

        # Simply overwrite oldest memories as we loop around.
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1  # Note, mem_cntr goes up forever! index gets modulo'd.


    def sample_buffer(self, batch_size):

        # If we haven't filled up the memory buffer, we only want to sample up to mem_ctr.
        # If we have filled up memory, we want to sample from all of it.
        #   np.random.choice uses np.arange, whose upper bound is exclusive, so no
        #   need for self.mem_size - 1, just self.mem_size.
        max_mem = min(self.mem_counter, self.mem_size)
        
        batch_indices = np.ones(batch_size, dtype=np.uint32) * -1  # * -1  so no possible indices will be in here.
        # print(batch_indices)
        for i in range(batch_size):
            while True:
                mRandInt = np.random.randint(max_mem)  # Allows to specify size=batch_size, but not replace=False.
                # print(mRandInt, end='\t')
                if mRandInt not in batch_indices:  # This should get EASIER as max_mem increases, not harder!
                    # print("Adding")
                    batch_indices[i] = int(mRandInt)
                    break  # the inner while loop, not the outer for loop.

        states = np.array([self.state_memory[index] for index in batch_indices])
        states_ = np.array([self.new_state_memory[index] for index in batch_indices])
        actions = self.action_memory[batch_indices]
        rewards = self.reward_memory[batch_indices]
        terminal = self.terminal_memory[batch_indices]

        return states, actions, rewards, states_, terminal
