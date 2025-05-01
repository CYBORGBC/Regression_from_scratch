import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Define the function and its gradient
def f(x):
    return x**2

def grad_f(x):
    return 2*x

# 2. Initialize
x = 8  # Starting point (far from minimum)
learning_rate = 0.2
n_iterations = 30

x_history = [x]

# 3. Perform Gradient Descent
for i in range(n_iterations):
    x = x - learning_rate * grad_f(x)
    x_history.append(x)

# 4. Prepare the plot
fig, ax = plt.subplots()
x_vals = np.linspace(-10, 10, 400)
y_vals = f(x_vals)

line, = ax.plot(x_vals, y_vals, 'b-')   # Function curve
point, = ax.plot([], [], 'ro')           # Moving point
ax.set_xlim(-10, 10)
ax.set_ylim(0, 100)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Gradient Descent Animation')

# 5. Animation function
def animate(i):
    point.set_data(x_history[i], f(x_history[i]))
    return point,

# 6. Create animation
ani = animation.FuncAnimation(fig, animate, frames=len(x_history), interval=300, repeat=False)

plt.show()
