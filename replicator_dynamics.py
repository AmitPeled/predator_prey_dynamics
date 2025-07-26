import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from model_functions import p_a, p_na  # assumes you have p_a() and p_na() defined
from dataclasses import dataclass


@dataclass
class Predator:
    p: float  # baseline success rates
    p_prime_base: float  # success rate when alerted (upper bound)
    p_min: float  # minimal success (floor) when x=1
    b: float  # steepness for p_prime(x)

    # Exponential form for p_prime(x)
    def p_prime(self, x):
        return (
                self.p_min +
                (self.p_prime_base - self.p_min) * (1 - math.exp(-self.b * x))
        )

    # Predator success f(x, p)
    def success_rate(self, x):
        return (1 - x) ** N * self.p + (1 - (1 - x) ** N) * self.p_prime(x)


# Parameters
N = 2
predator1 = Predator(p=0.9, p_prime_base=0.35, p_min=0.26, b=1)
predator2 = Predator(p=0.6, p_prime_base=0.35, p_min=0.26, b=1)
c = 2.0  # alerting penalty factor


# Payoffs for alerting vs silent
def P_alert(x, predator):
    return sum(
        math.comb(N - 1, i - 1) * x ** (i - 1) * (1 - x) ** (N - i)
        * p_a(i, N, predator.p_prime(x), c)
        for i in range(1, N + 1)
    )


def P_noalert(x, predator):
    return sum(
        math.comb(N - 1, i) * x ** i * (1 - x) ** (N - i - 1)
        * p_na(i, N, predator.p, predator.p_prime(x), c)
        for i in range(N)
    )


# ODE system
def system(vars, t, r=0.3, m=0.1):
    x, P1, P2 = vars
    total = max(P1 + P2, 1e-9)
    y1, y2 = P1 / total, P2 / total

    # Prey replicator
    average_predator = Predator(p=y1 * predator1.p + y2 * predator2.p,
                                p_prime_base=y1 * predator1.p_prime_base + y2 * predator2.p_prime_base,
                                p_min=y1 * predator1.p_min + y2 * predator2.p_min,
                                b=y1 * predator1.b + y2 * predator2.b)

    pa = P_alert(x, average_predator)
    pna = P_noalert(x, average_predator)
    avg_payoff = x * pa + (1 - x) * pna
    dxdt = x * (pa - avg_payoff)

    # Predator growth directly proportional to success rate
    dP1dt = r * P1 * predator1.success_rate(x) - m * P1
    dP2dt = r * P2 * predator2.success_rate(x) - m * P2

    return [dxdt, dP1dt, dP2dt]


# Time span
t = np.linspace(0, 5000, 20000)

# Initial conditions: list of (x0, P1_0, P2_0)
# inits = [(0.1,10,10), (0.5,10,10), (0.2,10,10)]
inits = [(0.5, 10, 10)]

fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for x0, P1_0, P2_0 in inits:
    sol = odeint(system, [x0, P1_0, P2_0], t)
    x_t, P1_t, P2_t = sol.T
    ax[0].plot(t, x_t, label=f"x₀={x0}")
    ax[1].plot(t, P1_t, label=f"P1 (init {P1_0}, x₀={x0})")
    ax[1].plot(t, P2_t, '--', label=f"P2 (init {P2_0}, x₀={x0})")
    ax[2].plot(t, P1_t / (P1_t + P2_t), label=f"Predator 1 frac, x₀={x0}")
    ax[2].plot(t, P2_t / (P1_t + P2_t), '--', label=f"Predator 2 frac, x₀={x0}")

ax[0].set_ylabel("Alert Frequency x");
ax[0].legend();
ax[0].grid(True)
ax[1].set_ylabel("Predator Populations");
ax[1].legend();
ax[1].grid(True)
ax[2].set_ylabel("Predator Fractions");
ax[2].set_xlabel("Time");
ax[2].legend();
ax[2].grid(True)

plt.tight_layout()
plt.show()
