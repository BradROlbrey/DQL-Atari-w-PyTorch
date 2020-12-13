
import sys
import signal
from time import sleep
from itertools import count
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch as T

from agent import DQNAgent
from environment import make_env
from plotter import plot_learning_curve


EXITTED = False  # For clean exit.
def signal_handler(sig, frame):
    print("Exiting")
    global EXITTED
    EXITTED = True
signal.signal(signal.SIGINT, signal_handler)


if __name__ == '__main__':

    env_name = 'BreakoutDeterministic-v4'
    # env_name = 'SpaceInvadersDeterministic-v4'
    # env_name = 'PongDeterministic-v4'
    # env_name = 'StarGunnerDeterministic-v4'
    env = make_env(env_name=env_name)

    print(env.observation_space)

    # Manually configure action mapping for each game because openaigym provides options!
    #   0: NOOP
    #   1: FIRE
    #   2: RIGHT
    #   3: LEFT
    #   4: RIGHTFIRE
    #   5: LEFTFIRE
    if 'Breakout' in env_name:
        actions = (0, 2, 3)
        fire_first = True  # We can fire immediately after losing a life.
    if 'SpaceInvaders' in env_name:
        # actions = (0, 1, 2, 3)
        actions = (2, 3, 4, 5)
            # Basically gives us the same possible behavior with less actions.
            # Move without and without firing. Staying in place and doing each
            #   can be emulated reasonable well by alternating 2&3 and 4&5.
        fire_first = False
    if 'Pong' in env_name:
        actions = (0, 4, 5)
        fire_first = False  # 'FIRELEFT' and 'FIRERIGHT' cover this.
    if 'StarGunner' in env_name:
        actions = list(range(18))
        fire_first = False

    # Get max lives in this environment
    env.reset()
    _,_,_,info = env.step(0)
    max_lives = info['ale.lives']
    
    best_avg_score = -np.inf

    agent_steps = 1000000  # epsilon_steps and mem_size
    agent = DQNAgent(
        env_name=env_name,
        n_actions=len(actions),
        input_dims=env.observation_space.shape,

        optimizer="RMSProp",
        double=False,

        mem_size=agent_steps,
        epsilon_steps=agent_steps,
        
        versioning=True,
    )
    save_version_interval = 10000
        # Save a new version of the model every x learn steps to be evaluated later.
        
    print(agent.representation)
    log_file = open(f'logs/{agent.name}.txt', 'w')
    log_file.write(agent.representation + '\n')
    figure_file = f'plots/{agent.name}.svg'

    n_steps = 0
    start_learning = (agent_steps // 20) - 1
    render_to_screen = False  # By default; actually significantly faster.

    output_steps = 0  # For plot and file logging
    scores, eps_history, steps_array = [], [], []
    log_output = []

    print(f"Start learning {start_learning}, version interval {save_version_interval}\n")

    i = -1
    while not EXITTED:
        i += 1
        
        done = False  # Environment resets when True.
        observation = env.reset()

        if 'StarGunner' in env_name:
            for j in range(70):  # The first 70 frames are menu title :p
                observation,_,_,_ = env.step(0)
        
        # FIRE first. Only necessary with Breakout.
        if fire_first:
            # Fire the ball!
            observation,_,_,_ = env.step(1)  # 1 is always 'FIRE'
        
        # env.render()  # This one doesn't matter tbh
        if i % 3 == 0:
            # Save the file i/o to every few games instead of every step.
            render_to_screen = open('render_to_screen.txt', 'r').readline()

        pseudo_done = False  # Agent receives this. True when life is lost, otherwise = done.
        num_lives = max_lives

        score = 0
        while not done and not EXITTED:
            action = agent.choose_action(observation)   # agent stores this action.
            mapped_action = actions[action]             # env steps with this mapped action.
            # print(action, mapped_action)
            observation_, reward, done, info = env.step(mapped_action)
            score += reward

            if render_to_screen:
                env.render()

            # SOFT RESET
            # The agent has no other way to understand that losing a life is costly.
            # We don't reset the environment though, we simply "tell" the agent it goofed.
            pseudo_done = done
            if info['ale.lives'] < num_lives:  # Always 0 with Pong.
                pseudo_done = True  # q_target is set to 0 in learn(). reward is 0.
                num_lives = info['ale.lives']
            # print(done, num_lives, pseudo_done)

            reward = np.sign(reward)  # Reward clipping
            agent.store_memory(observation, action, reward, observation_, pseudo_done)
            observation = observation_

            # Must store observation before doing FIRE first, if we are in fact firing first.
            # Only necessary with Breakout.
            if fire_first and pseudo_done:
                observation,_,_,_ = env.step(1)  # 1 is 'FIRE'

            if n_steps > start_learning:
                agent.learn()

            # Save models after we've committed to exploiting, regardless of score.
            if agent.versioning  and  n_steps >= agent_steps  and  n_steps % save_version_interval == 0:
                version = (n_steps - agent_steps) // save_version_interval
                # print(n_steps, n_steps - agent_steps, (n_steps - agent_steps) % version_interval, version)
                suffix = f'{np.mean(scores[-20:]):.2f}'
                agent.save_models(f'v{version:03}', suffix)  # Will be properly evaluated by another program later!

            n_steps += 1

            # sleep(1/30)
        output_steps += 1  # Every (10) games.

        scores.append(score)


        avg_score = np.mean(scores[-20:])  # Reduces our chance of saving a worse model as we descend after a steady climb.
        if avg_score > best_avg_score: best_avg_score = avg_score
        output = f'{i:>4} score {int(score):5}    average_score {avg_score:>7.1f}    best_avg_score {best_avg_score:>7.1f}   ' + \
                 f'epsilon {agent.epsilon:.3f}    steps {n_steps:>6}    time {datetime.now().strftime("%H:%M:%S")}'

        memory_summary = T.cuda.memory_summary(agent.q_train.device, abbreviated=True).splitlines()[3:8]
        memory_summary.insert(0, '_' * len(memory_summary[0]))
        memory_summary.append(' ' * len(memory_summary[0]))
        memory_summary = '\n'.join(memory_summary)
        if i != 0:
            sys.stdout.write("\x1B[A" * (memory_summary.count('\n')+1+2) + '\r')
                                                                    # The extra +2 is for the agent name.

        print(output)
        log_output.append(output)

        print(' ' * 117)
        print(agent.name)
        
        print(memory_summary)

        # Plot and log stuff
        eps_history.append(agent.epsilon)
        steps_array.append(n_steps)
        # This is kind of costly, so... do it less often!
        if output_steps % 10 == 0  and  n_steps > start_learning:
            plot_learning_curve(steps_array, scores, eps_history, figure_file)
            log_file.write('\n'.join(log_output) + '\n')
            log_file.flush()
            log_output.clear()
    env.close()
