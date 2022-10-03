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
u = 1 + X**2 + 2 * Y**2

print(u)
levels = np.linspace(1, 4, nx*2)
plt.contourf(u,200, levels = levels , cmap='jet')
plt.axis('off')
cbar = plt.colorbar()
cbar.set_label("$u$")
cbar.set_ticks(np.linspace(1,4,7))
plt.savefig("u_analytical.pdf")
plt.show()



