import numpy as np
import matplotlib.pyplot as plt

nx, ny = 100, 100

L = 1

x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)

X, Y = np.meshgrid(x, y)

# This is the solution of -/\u = 0 such that
# u(x=0) = 0
# u(x=L) = 0
# u(y=0) = 0
# u(y=L) = 0

u = 1 + X**2 + 2 * Y**2

fig, ax = plt.subplots(figsize=(8, 6)) # Use a specific size for better presentation
levels = np.linspace(1, 4, nx*2)
plt.contourf(u,200, levels = levels , cmap='jet')
plt.axis('off')
cbar = plt.colorbar()
cbar.set_label("$u$", fontsize=22)
cbar.set_ticks(np.linspace(1,4,7))
cbar.ax.tick_params(labelsize=16) # Adjust tick font size
plt.tight_layout()
plt.savefig("ResultsDir/u_analytical.pdf")
plt.savefig("ResultsDir/plots/u_analytical.png")
plt.show()