import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# parameters
g = 9.81
L1, L2 = 2.0, 1.5
m1, m2 = 1.0, 1.0
dt = 0.02

# initial state: [theta1, omega1, theta2, omega2]
state = np.array([np.pi/2, 0.0, np.pi/2 + 0.01, 0.0])

# physics 
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

# plot setup 
fig, (ax, ax_energy) = plt.subplots(2,1, figsize=(6,8), gridspec_kw={'height_ratios':[3,1]})
plt.subplots_adjust(left=0.1, bottom=0.35)

# pendulum plot
ax.set_aspect('equal')
ax.set_xlim(-L1-L2-0.5, L1+L2+0.5)
ax.set_ylim(-L1-L2-0.5, 0.5)
ax.set_title("Interactive Double Pendulum")

line1, = ax.plot([], [], lw=3, color='blue')
line2, = ax.plot([], [], lw=3, color='green')
bob1, = ax.plot([], [], 'ro', markersize=10)
bob2, = ax.plot([], [], 'yo', markersize=10)
trail, = ax.plot([], [], color='orange', lw=1)
trail_x, trail_y = [], []

# energy plot
ax_energy.set_xlim(0, 1000)
ax_energy.set_ylim(-5, 50)
ax_energy.set_title("Energies")
line_T, = ax_energy.plot([], [], color='r', label='Kinetic')
line_V, = ax_energy.plot([], [], color='b', label='Potential')
line_E, = ax_energy.plot([], [], color='k', label='Total')
ax_energy.legend()
energy_history_T, energy_history_V, energy_history_E = [], [], []

# sliders 
ax_theta1 = plt.axes([0.15, 0.25, 0.7, 0.03])
ax_theta2 = plt.axes([0.15, 0.20, 0.7, 0.03])
ax_L1 = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_L2 = plt.axes([0.15, 0.10, 0.7, 0.03])
ax_m1 = plt.axes([0.15, 0.05, 0.3, 0.03])
ax_m2 = plt.axes([0.55, 0.05, 0.3, 0.03])

slider_theta1 = Slider(ax_theta1, 'Theta1', 0, 2*np.pi, valinit=np.pi/2)
slider_theta2 = Slider(ax_theta2, 'Theta2', 0, 2*np.pi, valinit=np.pi/2+0.01)
slider_L1 = Slider(ax_L1, 'L1', 0.5, 3.0, valinit=L1)
slider_L2 = Slider(ax_L2, 'L2', 0.5, 3.0, valinit=L2)
slider_m1 = Slider(ax_m1, 'M1', 0.1, 5.0, valinit=m1)
slider_m2 = Slider(ax_m2, 'M2', 0.1, 5.0, valinit=m2)

def reset_sim(val):
    global state, trail_x, trail_y, L1, L2, m1, m2, energy_history_T, energy_history_V, energy_history_E
    L1 = slider_L1.val
    L2 = slider_L2.val
    m1 = slider_m1.val
    m2 = slider_m2.val
    state = np.array([slider_theta1.val, 0.0, slider_theta2.val, 0.0])
    trail_x.clear(); trail_y.clear()
    energy_history_T.clear(); energy_history_V.clear(); energy_history_E.clear()

for s in [slider_theta1, slider_theta2, slider_L1, slider_L2, slider_m1, slider_m2]:
    s.on_changed(reset_sim)

# animation 
def update(frame):
    global state, trail_x, trail_y, energy_history_T, energy_history_V, energy_history_E
    state = rk4(state, dt, L1, L2, m1, m2)
    (x1, y1), (x2, y2) = get_positions(state, L1, L2)
    line1.set_data([0, x1], [0, y1])
    line2.set_data([x1, x2], [y1, y2])
    bob1.set_data([x1], [y1])
    bob2.set_data([x2], [y2])

    # trail
    trail_x.append(x2)
    trail_y.append(y2)
    if len(trail_x) > 500:
        trail_x.pop(0); trail_y.pop(0)
    trail.set_data(trail_x, trail_y)

    # energies
    T, V, E = energies(state, L1, L2, m1, m2)
    energy_history_T.append(T)
    energy_history_V.append(V)
    energy_history_E.append(E)
    max_len = 500
    energy_history_T = energy_history_T[-max_len:]
    energy_history_V = energy_history_V[-max_len:]
    energy_history_E = energy_history_E[-max_len:]
    x_vals = np.arange(len(energy_history_T))
    line_T.set_data(x_vals, energy_history_T)
    line_V.set_data(x_vals, energy_history_V)
    line_E.set_data(x_vals, energy_history_E)
    ax_energy.set_xlim(0, max_len)

    return line1, line2, bob1, bob2, trail, line_T, line_V, line_E


def on_resize(event):
    fig.canvas.draw()

fig.canvas.mpl_connect('resize_event', on_resize)


ani = FuncAnimation(fig, update, frames=10000, interval=dt*1000, blit=True)

plt.show()
