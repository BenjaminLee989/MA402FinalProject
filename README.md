# MA402FinalProject
2D Cavity Flow Using Navier-Stokes Equation

## Problem Description

This project implements a lid-driven cavity flow solver using PETSC's nonlinear solver (SNES) through the petsc4py Python interface. The lid-driven cavity set-up is a foundation problem in the study of computational fluid dynamics (CFD)

The physical setup of this problem is a square cavity filled with viscous fluid (i.e. no inviscid assumptions) where the top wall moves horizontally at a constant velocity. ALl other walls are stationary to enforce the no-slip condition. The fluid is also treated as incompressible.

The solver computes the steady-state solution to the 2D incompressible Navier-Stokes equations:

$$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \frac{1}{Re}\nabla^2 u$$

$$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \frac{1}{Re}\nabla^2 v$$

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

## AI Translation Experience

This project was developed with the assistance of AI tools (Claude/Gemini) to translate the PETSc C tutorial code into a working petsc4py Python script. Overall, this project could be treated as a success. The AI tools correctly identified the overal structure of the SNES solver setup. The basic finite-difference stencil translation from C to Python was largely correct. Furthermore, the boundary conditions established by the AI were accurate and served as a foundational basis for the rest of the problem going forward. However, despite the many successes associated with the use of AI in this project, there were just as many, if not more, shortcomings. The AI repeatedly generated incorrect function calls, hallucinating variables in an attempt to preserve conformit between C and Python. Furthermore, the AI greatly struggled with properly calling the Jacobian function and it was a common point of failure across multiple builds of the code. As an aside, the AI was particularly helpful when figuring out how to run my code in VS code on Windows. For someone who has had most of their coding experience in MATLAB, one of the most difficult parts of this assignment for me was actually setting up the proper virtual environments on my computer so that everything could run smoothly.



