import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.linalg as la
import numpy.random as rn


class JumpProcessMasterEquation:
    def __init__(
        self, N: int, initial_dist: np.ndarray, transition_matrix: np.ndarray, M: int
    ):
        """initialisation of class

        Args:
            N (int): number of states in the system
            initial_dist (np.ndarray): initial distribution of the system
            transition_matrix (np.ndarray): 2D array where entry [a, b] (a neq b) is the transition rate from a to b and [a, a] is the exit rate = -sum of all [a, b] terms
            M (int): resolution of time we choose
        """
        self.N = N
        self.states = np.arange(1, N + 1)
        self.initial_dist = initial_dist
        self.transition_matrix = transition_matrix
        self.M = M
        self.dist_matr = np.zeros((N, N))

    def distance_matrix(self, d: np.ndarray):
        self.dist_matr = d

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
        propagator = la.expm(self.transition_matrix * t)

        return np.matmul(propagator.T, self.initial_dist)

    def empirical_probability(self, t: float, sample_size: int):
        """Cakculating empirical probability from a set of sample_size trajectories spanning from 0 to t. We consider a discretisation of N intermediate steps

        Args:
            t (float): final time
            M (int): Number of discretisation steps
            sample_size (int): Number of trajectories

        Returns:
            np.ndarray: probability matrix, entry [a,b] is probability of state b at timestep a
            np.ndarray: array consisting of times at which probability is calculated
        """
        p = np.zeros((self.M, self.N))

        dt = t / self.M

        time = np.arange(0, self.M) * dt

        for i in range(sample_size):
            trajectory, trajectory_time = self.Trajectory_sample(t)
            # print(trajectory, trajectory_time)
            index = 0
            for k in range(self.M):
                for j in range(len(trajectory_time)):
                    # print(time[k], trajectory_time[j], len(trajectory_time))
                    if time[k] < trajectory_time[j]:
                        index = j - 1
                        break

                # print(index, trajectory[index] - 1)
                p[k, trajectory[index] - 1] += 1

        self.p = p / sample_size
        self.time = time
        return p / sample_size, time

    def Z(self):

        z = np.zeros((self.N, self.N, self.M))

        for m in range(self.M):

            for k in range(self.N):
                for j in range(k, self.N):
                    if self.transition_matrix[k, j] > 0:
                        z[k, j, m] = (
                            self.transition_matrix[k, j] * self.p[m, k]
                            - self.transition_matrix[j, k] * self.p[m, j]
                        ) / (
                            self.transition_matrix[k, j] * self.p[m, k]
                            + self.transition_matrix[j, k] * self.p[m, j]
                        )
                        z[j, k, m] = -z[k, j, m]

        self.Z_matrix = z
        return z

    def current_full(self, t: float, N: int):
        return 0


# J = JumpProcess_description_MasterEquation(3, np.array([0, 0, 1]), np.zeros((3, 3)))
# print(J.initial_state(), J.states)


# example with a ring, say 10 states, with a biased of going "right"
N = 10
M = 50
p = np.zeros(N)
p[0] = 1

transition_matrix = np.zeros((N, N))

for i in range(N):

    transition_matrix[i, (i + 1) % N] = 0.7

    transition_matrix[i, i - 1] = 0.3

    transition_matrix[i, i] = -np.sum(transition_matrix[i, :])


T = JumpProcessMasterEquation(N, p, transition_matrix, M)
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


# master equation solution for state 1

t_final = 40
dtau = t_final / M
p_me = []
p_me_4 = []
p_me_10 = []
for m in range(M):
    ME = T.master_eq_evolution(m * dtau)
    p_me.append(ME[0])
    p_me_4.append(ME[3])
    p_me_10.append(ME[-1])

prob, time = T.empirical_probability(t_final, 3000)
z = T.Z()
print(prob, time)
print(z[:, :, 1])


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
ax[0].plot(time, prob[:, 0], label=r"$p_1(\tau)$, empirical")
ax[0].plot(time, prob[:, 3], label=r"$p_4(\tau)$, empirical")
ax[0].plot(time, prob[:, -1], label=r"$p_{10}(\tau)$, empirical")
ax[0].grid()
ax[0].legend()
ax[0].set_xlabel(r"$\tau$")
ax[0].set_ylabel(r"$p_i(\tau)$")

ax[1].plot(time, p_me, label=r"$p_1(\tau)$, master eq")
ax[1].plot(time, p_me_4, label=r"$p_4(\tau)$, master eq")
ax[1].plot(time, p_me_10, label=r"$p_{10}(\tau)$, master eq")
ax[1].grid()
ax[1].legend()
ax[1].set_xlabel(r"$\tau$")


plt.show()
