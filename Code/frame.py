import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import numpy.random as rn


class JumpProcessMasterEquation:
    def __init__(
        self, N: int, initial_dist: np.ndarray, transition_matrix: np.ndarray
    ) -> None:
        self.N = N
        self.states = np.arange(1, N + 1)
        self.initial_dist = initial_dist
        self.transition_matrix = transition_matrix

    def initial_state(self):
        """Function to randomly determine initial condition given a initial distribution

        Returns:
            int: (random) initial state
        """
        n_0 = rn.choice(self.states, 1, p=self.initial_dist)

        return n_0[0]

    def Trajectory_sample(self, final_time: float):

        # list with states in it
        trajectory = [self.initial_state()]
        # list of transition times to the next respective states
        state_time = [0]
        # print(self.transition_matrix[trajectory[-1]-1, trajectory[-1]-1])
        # start at time 0
        tau = 0
        while tau <= final_time:
            # draw the escape time according to an exponential distribution
            # note that the index for states starts at 1, hence we have to subtract to get the appropriate index
            tau += rn.exponential(
                scale=-1
                / self.transition_matrix[trajectory[-1] - 1, trajectory[-1] - 1]
            )
            if tau <= final_time:
                state_time.append(tau)
                # figure out which state it transitions to
                transition_dist = self.transition_matrix[trajectory[-1] - 1] / (
                    -self.transition_matrix[trajectory[-1] - 1, trajectory[-1] - 1]
                )
                transition_dist[trajectory[-1] - 1] = 0
                trajectory.append(rn.choice(self.states, 1, p=transition_dist)[0])
        state_time.append(final_time)
        return trajectory, state_time

    def master_eq_evolution(self, t):
        """function to evolve probabilities to a certain time according to the master equation
        Note that this is the 'actual' probability, not the empiric sampling probability

        Args:
            t (float): time to propagate to

        Returns:
            np.ndarray: probability at time t
        """
        propagator = sc.linalg.expm(self.transition_matrix * t)

        return np.matmul(propagator, self.initial_dist)

    def empirical_probability(self, t: float, N: int, sample_size: int):

        p = np.zeros((N, self.N))

        dt = t / N

        time = np.arange(0, N) * dt

        for i in range(sample_size):
            trajectory, trajectory_time = self.Trajectory_sample(t)
            # print(trajectory, trajectory_time)
            index = 0
            for k in range(N):
                for j in range(len(trajectory_time)):
                    # print(time[k], trajectory_time[j], len(trajectory_time))
                    if time[k] < trajectory_time[j]:
                        index = j - 1
                        break

                # print(index, trajectory[index] - 1)
                p[k, trajectory[index] - 1] += 1

        return p / sample_size, time


# J = JumpProcessMasterEquation(3, np.array([0, 0, 1]), np.zeros((3, 3)))
# print(J.initial_state(), J.states)


# example with a ring, say 10 states, with a biased of going "right"
N = 10
p = np.zeros(N)
p[0] = 1

transition_matrix = np.zeros((N, N))

for i in range(N):

    transition_matrix[i, (i + 1) % N] = 0.7

    transition_matrix[i, i - 1] = 0.3

    transition_matrix[i, i] = -np.sum(transition_matrix[i, :])


T = JumpProcessMasterEquation(N, p, transition_matrix)
tra = T.Trajectory_sample(10)


def trajectory_plot(states, transitions, t):
    fig = plt.figure()

    for i in range(len(states) - 1):
        plt.plot([transitions[i], transitions[i + 1]], [states[i], states[i]], "r-")
        plt.plot(
            [transitions[i + 1], transitions[i + 1]], [states[i], states[i + 1]], "r-"
        )
    plt.plot([transitions[-2], transitions[-2]], [states[-2], states[-1]], "r-")
    plt.plot([transitions[-2], transitions[-1]], [states[-1], states[-1]], "r-")

    plt.savefig("test.pdf", dpi=500, bbox_inches="tight")


# trajectory_plot(tra[0], tra[1], 10)


prob, time = T.empirical_probability(50, 20, 100)
print(prob, time)


fig = plt.figure()
plt.plot(time, prob[:, 0])
plt.plot(time, np.ones_like(time) * 0.1)
plt.show()
