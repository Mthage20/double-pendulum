import numpy as np

# --- parameters ---
g = 9.81
L1, L2 = 2.0, 1.5
m1, m2 = 1.0, 1.0
dt = 0.02

# initial state: [theta1, omega1, theta2, omega2]
state = np.array([np.pi/2, 0.0, np.pi/2 + 0.01, 0.0])

# --- physics ---
def derivs(s, L1, L2, m1, m2):
    θ1, ω1, θ2, ω2 = s
    Δ = θ1 - θ2
    M = m1 + m2
    den = M - m2*np.cos(Δ)**2
    α1 = (-np.sin(Δ) * (m2*L2*ω2**2 + m2*L1*ω1**2*np.cos(Δ))
          - g*(M*np.sin(θ1) - m2*np.sin(θ2)*np.cos(Δ))) / (L1 * den)
    α2 = ( np.sin(Δ) * (M*L1*ω1**2 + m2*L2*ω2**2*np.cos(Δ))
           + g*(M*np.sin(θ1)*np.cos(Δ) - M*np.sin(θ2)) ) / (L2 * den)
    return np.array([ω1, α1, ω2, α2])

def rk4(s, dt, L1, L2, m1, m2):
    k1 = derivs(s, L1, L2, m1, m2)
    k2 = derivs(s + 0.5*dt*k1, L1, L2, m1, m2)
    k3 = derivs(s + 0.5*dt*k2, L1, L2, m1, m2)
    k4 = derivs(s + dt*k3, L1, L2, m1, m2)
    return s + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def get_positions(s, L1, L2):
    θ1, _, θ2, _ = s
    x1 = L1 * np.sin(θ1)
    y1 = -L1 * np.cos(θ1)
    x2 = x1 + L2 * np.sin(θ2)
    y2 = y1 - L2 * np.cos(θ2)
    return (x1, y1), (x2, y2)

def energies(s, L1, L2, m1, m2):
    θ1, ω1, θ2, ω2 = s
    x1, y1 = L1*np.sin(θ1), -L1*np.cos(θ1)
    x2, y2 = x1 + L2*np.sin(θ2), y1 - L2*np.cos(θ2)
    vx1, vy1 = L1*ω1*np.cos(θ1), L1*ω1*np.sin(θ1)
    vx2 = vx1 + L2*ω2*np.cos(θ2)
    vy2 = vy1 + L2*ω2*np.sin(θ2)
    T = 0.5*m1*(vx1**2 + vy1**2) + 0.5*m2*(vx2**2 + vy2**2)
    V = m1*g*y1 + m2*g*y2
    return T, V, T+V
