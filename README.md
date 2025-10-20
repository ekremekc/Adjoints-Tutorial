# Adjoints-tutorial

In this tutorial repository, we aim to show how Poisson mother problem is adjointized and solver. We aim to provide real-life application to demonstrate the applicability of adjoint solvers for classical engineering problems.

# Installation 

This tutorial uses finite element method to solve Poisson equations. Hence, we use open-source FEniCSx for this. One of the easiest way to install FEniCSx is to use conda enviroments. We provide step-by-step commands below to generate a conda environment. It is recommended to use VSCode as an IDE.

## Conda

This library can be installed directly with conda. We can make a conda environment for FEniCSx and then install this library into it.

```bash
conda create -n adjoint-control python=3.11.0
conda activate adjoint-control
conda install pip
conda install -c conda-forge fenics-dolfinx=0.9.0 pyvista=0.44.1 # Linux and macOS
```

If you exit the environment for some reason, we can re-activate it by:

```bash
conda activate adjoint-control
```