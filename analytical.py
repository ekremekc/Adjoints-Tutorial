import numpy as np
import matplotlib.pyplot as plt

nx = 100
ny = 100

L = 1

x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)


X , Y = np.meshgrid(x, y)

# This is the solution of -/\u = 0 such that
# u(x=0) = 0
# u(x=L) = 0
# u(y=0) = 0
# u(y=L) = 0

n = 1
u = np.sin(n * np.pi * Y / L) * np.sin(n * np.pi * X / L)

plt.contourf(u,200)
plt.show()


