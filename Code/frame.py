import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sc
import networkx as nx
import my_networkx as my_nx
import scipy.linalg as la
import numpy.random as rn
from matplotlib.patches import Arc, RegularPolygon
from numpy import radians as rad

class JumpProcessMasterEquation:
    def __init__(
        self, N: int, initial_dist: np.ndarray, transition_matrix: np.ndarray, M: int
    ):
        """initialisation of class

        Args:
            N (int): number of states in the system
            initial_dist (np.ndarray): initial distribution of the system
            transition_matrix (np.ndarray): 2D array where entry [b, a] (a neq b) is the transition rate from a to b and [a, a] is the exit rate = -sum of all [a, b] terms
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
                transition_dist = self.transition_matrix[:, trajectory[-1] - 1] / (
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

        return np.matmul(propagator, self.initial_dist)

    def empirical_probability(self, t: float, traj, traj_time):
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
        sample_size = len(traj)

        for i in range(sample_size):
            # trajectory, trajectory_time = self.Trajectory_sample(t)
            # print(trajectory, trajectory_time)
            index = 0
            for k in range(self.M):
                for j in range(len(traj_time[i])):
                    # print(time[k], trajectory_time[j], len(trajectory_time))
                    if time[k] < traj_time[i][j]:
                        index = j - 1
                        break

                # print(index, trajectory[index] - 1)
                p[k, traj[i][index] - 1] += 1

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
                            self.transition_matrix[j, k] * self.p[m, k]
                            - self.transition_matrix[k, j] * self.p[m, j]
                        ) / (
                            self.transition_matrix[j, k] * self.p[m, k]
                            + self.transition_matrix[k, j] * self.p[m, j]
                        )
                        z[j, k, m] = -z[k, j, m]

        self.Z_matrix = z
        return z

    def current_full(self, t: float, N: int):
        return 0







def k_singlecycle(A: np.ndarray, E: np.ndarray, cyc: np.ndarray, scaling: float, direction: int):
    """Function to determine the forward and backward cycle pertubation coefficients

    Args:
        A (np.ndarray): Adjacency matrix. 
        E (np.ndarray): Energies.
        cyc (np.ndarray): Cycle array consisting of all vertices in the cycle (start/end point included).

    Returns:
        np.ndarray: Matrix containing the pertubation coefficients
    """
    N = len(E)
    k_matrix = np.zeros((N, N))

    prop = []
    for i in range(len(cyc)):
        prop.append(np.exp(-(E[cyc[i-1]-1] + E[cyc[i]-1])/2))

    k_value = scaling * min(prop) * direction

    for i in range(len(cyc)):
        k_matrix[cyc[i]-1, cyc[i-1]-1] = k_value
        k_matrix[cyc[i-1]-1, cyc[i]-1] = -k_value
    
    return k_matrix

def Independent(c1: np.ndarray, c2: np.ndarray):
    # get intersection between cycles
    intersect = np.intersect1d(c1, c2)
    # size of overlap
    D = len(intersect)
    # size of cycles
    l1 = len(c1)
    l2 = len(c2)
    # if there is only up to single point in intersection, they are independent
    if D < 2:
        return True, [], []
    # need to check if there are multiple points in overlap
    else:
        
        # list of intersection links
        Inter = []
        # list for orientation of parallel segments. If they are parallel, eg both contain 1~2, then add 1, 
        # if, on the other hand, they are anti-parallel, eg cycle 1 contains 1~2 but cycle 2 contains 2~1, 
        # we add a -1
        orientation = []
        for d in range(D):
            # index for d-th overlap element in cycle 1 and 2
            i1 = np.argwhere(c1 == intersect[d])[0, 0]
            i2 = np.argwhere(c2 == intersect[d])[0, 0]
            if c1[(i1 + 1)%l1] == c2[(i2 + 1)%l2] and c2[(i2 + 1)%l2] in intersect:
                Inter.append([c1[i1], c1[(i1+1)%l1]])
                orientation.append(1)
            elif c1[(i1 + 1)%l1] == c2[(i2 - 1)%l2] and c2[(i2 - 1)%l2] in intersect:
                Inter.append([c1[i1], c1[(i1+1)%l1]])
                orientation.append(-1)
            
            

        if len(Inter) == 0:
            print(Inter, orientation)
            return True, [], []
        else:
            return False, Inter, orientation

def k_multicycle(A: np.ndarray, E: np.ndarray, cyc: np.ndarray, scaling: np.ndarray, direction: np.ndarray):
    
    # size of network
    N = len(E)
    # final pertubation matrix
    k_matrix = np.zeros((N, N))

    # make 3d matrix with entries of possible k values
    k_prop_matrix = np.zeros((len(cyc), N, N))
    
    # find possible value of k in each cycle
    prop_k_val = []
    for i in range(len(cyc)):
        possible_vals = []
        for j in range(len(cyc[i])):
            possible_vals.append(np.exp(-(E[cyc[i][j]-1] + E[cyc[i][j-1]-1])/2))
        min_val = min(possible_vals)*scaling[i]*direction[i]
        prop_k_val.append(min_val)

        for j in range(len(cyc[i])):
            k_prop_matrix[i, cyc[i][j]-1, cyc[i][j-1]-1] = min_val
            k_prop_matrix[i, cyc[i][j-1]-1, cyc[i][j]-1] = -min_val
    

    # determine dependent cycles
    for i in range(len(cyc)):
        for j in range(i, len(cyc)):
            if i != j:
                # check independence and get intersection and orientation
                check, intersection, orientation = Independent(cyc[i], cyc[j])
                # if they aren't independent, continue
                if not check:
                    for k in range(len(intersection)):
                        pertubation = np.abs(prop_k_val[i] + prop_k_val[j])
                        if pertubation >= np.exp(-(E[intersection[k][0]] + E[intersection[k][1]]) / 2):
                            raise ValueError("Pertubation to large for cycle %d and %d" % (i, j))
                    
    # If all works, we simply add the matrices to get the final pertubation matrix
    for i in range(len(cyc)):
        k_matrix += k_prop_matrix[i]


    return k_matrix, prop_k_val

def L_full(A: np.ndarray, E: np.ndarray, cyc: np.ndarray, scaling: np.ndarray, direction: np.ndarray):
    """_summary_

    Args:
        A (np.ndarray): Adjecency matrix, NxN. Needed to leave structure the same.
        E (np.ndarray): Energies of states, N. For invariant probability measure and transition rates.

    Returns:
        np.ndarray: Full master operator
        np.ndarray: Steady state probability vector
        np.ndarray: Antisymmetric part
        np.ndarray: Symmetric part
    """

    # size of system
    N = len(E)

    # Pertubation coefficient matrix
    k = np.zeros((N,N))

    if len(cyc) == 1:
        k = k_singlecycle(A, E, cyc[0], scaling, direction=direction)
    elif len(cyc) > 1:
        k = k_multicycle(A, E, cyc, scaling, direction)[0]

    L = np.zeros_like(A)

    # probability vector
    peq = np.exp(-E) / np.sum(np.exp(-E))

    # symmetric part of master operator
    Ls = np.zeros((N, N))
    # off-diagonal entries
    for i in range(N):
        for j in range(N):
            # check for edge
            if A[i, j] == 1:
                Ls[i, j] = np.exp((E[j] - E[i])/2)
                # Ls[j, i] = np.exp((E[i] - E[j])/2)
        
    # diagonal entries
    for i in range(N):
        Ls[i, i] = -np.sum(Ls[:, i])
    
    # antisymmetric part
    La = np.zeros((N, N))
    # only has off-diagonal entries
    for i in range(N):
        for j in range(N):
            # check for edge
            if A[i, j] == 1:
                La[i, j] = k[i, j]* np.exp(E[j])
    
    # full master operator is sum of symmetric and antisymmetric part    
    L = La + Ls

    return L, peq, La, Ls

def Edge_Affinity(L: np.ndarray, p: np.ndarray, ind_from: int, ind_to: int):
    """ Edge affinity for a given network with given probability measure

    Args:
        L (np.ndarray): Master operator
        p (np.ndarray): probability measure
        ind_from (int): index for starting state
        ind_to (int): index for target state

    Returns:
        float: affinity for edge
    """
    a = p[ind_from] * L[ind_to, ind_from] / (p[ind_to] * L[ind_from, ind_to])
    print(p[ind_from] * L[ind_to, ind_from], p[ind_to] * L[ind_from, ind_to], a, np.log(a))
    return np.log(p[ind_from] * L[ind_to, ind_from] / (p[ind_to] * L[ind_from, ind_to]))


def Cycle_Affinity(L: np.ndarray, p: np.ndarray, cyc: np.ndarray):

    A_tot = 0

    for i in range(len(cyc)):
        A_tot += Edge_Affinity(L, p, cyc[i-1] - 1, cyc[i] - 1)
    print("##########################################")
    return A_tot


def Entropy_Change(A: np.ndarray, L: np.ndarray, p: np.ndarray):
    N = len(p)
    Sigma = 0

    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                Sigma += p[i] * L[j, i] * np.log(p[i] * L[j, i] / p[j] * L[i, j])

    return Sigma


def drawCirc(ax,radius,centX,centY,angle_,theta2_,color_='black', direction=0):
    #========Line
    arc = Arc([centX,centY],radius,radius,angle=angle_,
          theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=10,color=color_)
    ax.add_patch(arc)
    #========Create the arrow head
    if direction == 0:   
        endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
        endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))
    elif direction == 1:
        endX=centX+(radius/2)*np.cos(rad(angle_)) #Do trig to determine end position
        endY=centY+(radius/2)*np.sin(rad(angle_))

    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            radius/9,                # radius
            rad(angle_+theta2_),     # orientation
            color=color_
        )
    )

def plot_network(A, ax_plot, we=np.ones((6, 6)), position={1:(0,0),2:(1,0),3:(2,1),4:(3,0),5:(2,-1),6:(2,0)}, name="figures/Example_Calmodulin_1.pdf", node_color=np.ones(6), edge_name="r", arrstyle="<|-", font_size=20, width_scale=100):
    
    G = nx.DiGraph() 
    edges = []
    k = 0
    color = []
    N = len(A[:,0])
    for i in range(len(A[:,0])): 
        for j in range(len(A[:,0])): 
            if we[i, j] > 1e-15 and A[i, j] == 1: 
                #G.add_edge(i,j)
                if edge_name == "r":
                    edges.append((i+1, j+1, {"weight":we[i,j], "w":r"$r_{%d%d}$" % (j+1, i+1)}))
                elif edge_name == "j":
                    #edges.append((i+1, j+1, {"weight":we[i,j], "w":r"$p_{%d}r_{%d%d}$" % (j+1, j+1, i+1)}))
                    edges.append((i+1, j+1, {"weight":we[i,j], "w":r"$J_{%d%d}$" % (j+1, i+1)}))
                    # print(i+1, j+1, we[i,j])
                elif edge_name == "s":
                    edges.append((i+1, j+1, {"weight":we[i,j], "w":""}))
                color.append(we[i,j])

        G.add_node(i+1)

    G.add_edges_from(edges)

    weights = [G[u][v]['weight'] for u,v in G.edges]
    
    width_weight = [w*width_scale for w in weights]

    fixed_positions = position #dict with two of the positions set
    fixed_nodes = fixed_positions.keys()
    pos = nx.spring_layout(G, pos=fixed_positions, fixed=fixed_nodes)

    cmap = plt.cm.plasma

    if edge_name == "r":
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 10))
        # plt.tight_layout()

        nx.draw_networkx_nodes(G, pos, ax=ax_plot, node_size=3000, edgecolors="k")
        nx.draw_networkx_labels(G, pos, ax=ax_plot, font_size=30)

        curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
        straight_edges = list(set(G.edges()) - set(curved_edges))
        
        nx.draw_networkx_edges(G, pos, ax=ax_plot, edgelist=straight_edges, arrowstyle="<|-", width=5, node_size=3000)
        arc_rad = 0.10
        edges_plot = nx.draw_networkx_edges(G, pos, ax=ax_plot, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', arrowstyle=arrstyle, width=weights, edge_color=weights, arrowsize=20, node_size=3000)

        pc = mpl.collections.PatchCollection(edges_plot, cmap=cmap)
        pc.set_array(weights)

        edge_weights = nx.get_edge_attributes(G,'w')
        curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
        straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
        my_nx.my_draw_networkx_edge_labels(G, pos, ax=ax_plot, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad, font_size=20)
        nx.draw_networkx_edge_labels(G, pos, ax=ax_plot, edge_labels=straight_edge_labels,rotate=False)
        plt.colorbar(pc, ax=ax_plot)
        
    elif edge_name == "j" or edge_name == "s":
        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 10), gridspec_kw={'width_ratios': [1, 3]})
        # plt.tight_layout()

        nx.draw_networkx_nodes(G, pos, ax=ax_plot, node_size=3000, edgecolors="k")
        nx.draw_networkx_labels(G, pos, ax=ax_plot, font_size=30)

        curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
        straight_edges = list(set(G.edges()) - set(curved_edges))        
        
        # Fix order of weigths... sets destroy order
        width_weight2 = []
        color_weight = []
        for i in range(len(straight_edges)):
            width_weight2.append(width_scale*G[straight_edges[i][0]][straight_edges[i][1]]["weight"])
            color_weight.append(G[straight_edges[i][0]][straight_edges[i][1]]["weight"])
        
        nx.draw_networkx_edges(G, pos, ax=ax_plot, edgelist=straight_edges, arrowstyle=arrstyle, width=width_weight2, edge_color=color_weight, arrowsize=20, node_size=3000)
        arc_rad = 0.15
        edges_plot = nx.draw_networkx_edges(G, pos, ax=ax_plot, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', arrowstyle="<|-", width=width_weight2, edge_color=color_weight, arrowsize=20, node_size=3000)
        
        pc = mpl.collections.PatchCollection(edges_plot, cmap=cmap)
        pc.set_array(weights)

        edge_weights = nx.get_edge_attributes(G,'w')
        curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
        straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
        my_nx.my_draw_networkx_edge_labels(G, pos, ax=ax_plot, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad, font_size=font_size)
        nx.draw_networkx_edge_labels(G, pos, ax=ax_plot, edge_labels=straight_edge_labels,rotate=False, font_size=font_size)
        plt.colorbar(pc, ax=ax_plot)
        
        #ax[0].bar(np.arange(1, N+1), node_color, width=np.ones(N)*0.8, color="green", align="center", edgecolor="black")
        #ax[0].set_xlabel(r"State $i$")
        #ax[0].set_ylabel(r"$p_i$")
        #ax[0].grid()
        #ax[0].set_xticks(np.arange(1, N+1))
        
    # plt.savefig(name, dpi=500, bbox_inches="tight")

def printMatrix(s):
    
    for i in range(len(s)):
        for j in range(len(s[0])):
            print("%.2f   " % (s[i][j]), end="")
        print("\n") 



# # J = JumpProcess_description_MasterEquation(3, np.array([0, 0, 1]), np.zeros((3, 3)))
# # print(J.initial_state(), J.states)


# # example with a ring, say 10 states, with a biased of going "right"
# N = 10
# M = 50
# p = np.zeros(N)
# p[0] = 1

# transition_matrix = np.zeros((N, N))

# for i in range(N):

#     transition_matrix[i, (i + 1) % N] = 0.7

#     transition_matrix[i, i - 1] = 0.3

#     transition_matrix[i, i] = -np.sum(transition_matrix[i, :])


# T = JumpProcessMasterEquation(N, p, transition_matrix, M)
# tra = T.Trajectory_sample(10)


# def trajectory_plot(states, transitions, t):
#     fig = plt.figure()

#     for i in range(len(states) - 1):
#         plt.plot([transitions[i], transitions[i + 1]], [states[i], states[i]], "r-")
#         plt.plot(
#             [transitions[i + 1], transitions[i + 1]], [states[i], states[i + 1]], "r-"
#         )
#     plt.plot([transitions[-2], transitions[-2]], [states[-2], states[-1]], "r-")
#     plt.plot([transitions[-2], transitions[-1]], [states[-1], states[-1]], "r-")

#     plt.savefig("test.pdf", dpi=500, bbox_inches="tight")


# # trajectory_plot(tra[0], tra[1], 10)


# # master equation solution for state 1

# t_final = 40
# dtau = t_final / M
# p_me = []
# p_me_4 = []
# p_me_10 = []
# for m in range(M):
#     ME = T.master_eq_evolution(m * dtau)
#     p_me.append(ME[0])
#     p_me_4.append(ME[3])
#     p_me_10.append(ME[-1])

# prob, time = T.empirical_probability(t_final, 3000)
# z = T.Z()
# print(prob, time)
# print(z[:, :, 1])


# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
# ax[0].plot(time, prob[:, 0], label=r"$p_1(\tau)$, empirical")
# ax[0].plot(time, prob[:, 3], label=r"$p_4(\tau)$, empirical")
# ax[0].plot(time, prob[:, -1], label=r"$p_{10}(\tau)$, empirical")
# ax[0].grid()
# ax[0].legend()
# ax[0].set_xlabel(r"$\tau$")
# ax[0].set_ylabel(r"$p_i(\tau)$")

# ax[1].plot(time, p_me, label=r"$p_1(\tau)$, master eq")
# ax[1].plot(time, p_me_4, label=r"$p_4(\tau)$, master eq")
# ax[1].plot(time, p_me_10, label=r"$p_{10}(\tau)$, master eq")
# ax[1].grid()
# ax[1].legend()
# ax[1].set_xlabel(r"$\tau$")


# plt.show()
