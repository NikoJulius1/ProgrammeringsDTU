# Code 13.1: Predator-Prey Update
class PredatorPrey:
    def __init__(self, r, K, a, D, s, m):
        self.r = r  # Growth rate of prey
        self.K = K  # Carrying capacity of prey
        self.a = a  # Predation rate
        self.D = D  # Half-saturation constant
        self.s = s  # Growth rate of predator
        self.m = m  # Mortality rate of predator

    def __str__(self):
        return f"r:{self.r}, K:{self.K}, a:{self.a}, D:{self.D}, s:{self.s}, m:{self.m}"

    def update(self, N, P):
        delta_N = self.r * N * (1 - N / self.K) - (self.a * P * N) / (N + self.D)
        delta_P = self.s * P * (1 - self.m * P / N)
        return N + delta_N, P + delta_P

# Example usage:
model = PredatorPrey(0.16, 2400, 5, 1000, 0.01, 2)
print(model)
N, P = 120, 40
N, P = model.update(N, P)
print(N, P)

# Code 13.2: Predator-Prey Evolution
import matplotlib.pyplot as plt

def plot_evolution(model, N0, P0, iterations):
    N, P = N0, P0
    N_values, P_values = [N], [P]

    for _ in range(iterations):
        N, P = model.update(N, P)
        N_values.append(N)
        P_values.append(P)

    plt.figure()
    plt.plot(N_values, P_values, label="Phase plot")
    plt.xlabel("Prey population")
    plt.ylabel("Predator population")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(range(iterations + 1), N_values, label="Prey")
    plt.plot(range(iterations + 1), P_values, label="Predator")
    plt.xlabel("Time (iterations)")
    plt.ylabel("Population")
    plt.legend()
    plt.show()

# Example usage:
plot_evolution(model, 120, 40, 1000)

# Code 13.3: Predator-Prey Module
# predator_prey.py
class PredatorPrey:
    def __init__(self, r, K, a, D, s, m):
        self.r = r
        self.K = K
        self.a = a
        self.D = D
        self.s = s
        self.m = m

    def update(self, N, P):
        delta_N = self.r * N * (1 - N / self.K) - (self.a * P * N) / (N + self.D)
        delta_P = self.s * P * (1 - self.m * P / N)
        return N + delta_N, P + delta_P

# predator_prey_simulation.py
from predator_prey import PredatorPrey
import matplotlib.pyplot as plt

model = PredatorPrey(0.16, 2400, 5, 1000, 0.01, 2)
plot_evolution(model, 120, 40, 1000)

# Problem 13.4: Allee Effect Analysis
class AlleeEffect:
    def __init__(self, r, K, A):
        self.r = r  # Growth rate
        self.K = K  # Carrying capacity
        self.A = A  # Allee threshold

    def __str__(self):
        return f"r:{self.r}, K:{self.K}, A:{self.A}"

    def update(self, N, dt):
        dN_dt = self.r * N * (1 - N / self.K) * ((N / self.A) - 1)
        return N + dN_dt * dt

    def evolve(self, N0, T, dt):
        times, populations = [0], [N0]
        N = N0
        for t in range(1, int(T / dt) + 1):
            N = self.update(N, dt)
            times.append(t * dt)
            populations.append(N)
        return times, populations

    def plot_evolutions(self, N0_list, T, dt):
        for N0 in N0_list:
            times, populations = self.evolve(N0, T, dt)
            plt.plot(times, populations, label=f"N0={N0}")
        plt.xlabel("Time")
        plt.ylabel("Population Size")
        plt.legend()
        plt.title("Allee Effect Dynamics")
        plt.show()

# Example usage:
allee_model = AlleeEffect(0.03, 9000, 1000)
allee_model.plot_evolutions([500, 1000, 5000, 7000], 100, 0.1)
