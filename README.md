# Adjoints-tutorial

In this tutorial repository, we show how Poisson mother problem is adjointized and solved. In addition, we aim to provide real-life application to demonstrate the applicability of adjoint solvers for classical engineering problems.

## Journal Article Information

The outputs and the purpose of the code is explained in detail in the manuscript published in [Journal of Mathematical Science and Modelling](https://dergipark.org.tr/en/pub/jmsm). The article can be accessed from [this link](https://doi.org/10.33187/jmsm.1810588). Further, the bibtex entry for citation is give below:

```
@article{ekici2026adjoint,
  author  = {Ekici, E.},
  title   = {Adjoint Optimization for the {Poisson} Problem with Applications},
  journal = {Journal of Mathematical Sciences and Modelling},
  year    = {2026},
  pages   = {77--84},
  doi     = {10.33187/jmsm.1810588},
  note    = {Advanced Online Publication}
}
```

# Installation 

This tutorial uses finite element method (FEM) to solve Poisson equations. Hence, we use open-source FEniCSx for using FEM. One of the easiest way to install FEniCSx is to use conda enviroments. We provide step-by-step commands below to generate a conda environment. It is recommended to use VSCode as an IDE.

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
