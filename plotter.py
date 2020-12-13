
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(steps, scores, epsilons, filename):
    average_over = 500
    mean, mean_plus, mean_minus = get_mean_stddev(scores, average_over)

    fig, ax = plt.subplots(1, 1)

    # Scores
    color = 'tab:blue'
    ax.set_ylabel(f'Average Score ({average_over} games)', color=color)
    ax.plot(steps, mean, color=color)
    ax.fill_between(steps, mean_minus, mean_plus, color=color, alpha=0.2)
    ax.tick_params(axis='y', labelcolor=color)
    
    # Epsilon, needs own y axis
    axE = ax.twinx()  # Instantiate a second axes that shares the same x-axis.
    axE.plot(steps, epsilons, color='k', alpha=0.2)
    axE.yaxis.set_visible(False)  # We don't want its y labeling

    fig.savefig(filename)


# Get the
#   1. mean
#   2. mean + std dev
#   3. mean - std dev
# of the past [count] games.
def get_mean_stddev(scores, count=500):

    mean =  [np.mean(  scores[max(0, t-count):(t+1)]  )   for t in range(len(scores))  ]

    # averages = []
    # for t in range(len(scores)):
    #     print(scores[max(0, t-count):(t+1)])
    #     averages.append(np.mean(  scores[max(0, t-count):(t+1)]  ))

    stddev = [np.std(  scores[max(0, t-count):(t+1)]  )   for t in range(len(scores))  ]

    # mean_plus =  [  mean[t] + stddev[t]   for t in range(len(scores))  ]
    # mean_minus = [  mean[t] - stddev[t]   for t in range(len(scores))  ]

    mean_plus = np.add(mean, stddev)
    mean_minus = np.subtract(mean, stddev)

    return mean, mean_plus, mean_minus
