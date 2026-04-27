"""
Lid-Driven Cavity Flow — 2D Incompressible Navier-Stokes
=========================================================

This module solves the steady 2D incompressible Navier-Stokes equations
for the canonical lid-driven cavity flow problem using PETSc via petsc4py.

Governing equations (non-dimensionalized):
    (u · ∇)u = -∇p + (1/Re) ∇²u      (momentum)
    ∇ · u = 0                           (continuity / incompressibility)

Domain:
    Unit square Ω = [0, 1] × [0, 1]

Boundary Conditions:
    - Top wall    (y = 1):  u = 1, v = 0  (moving lid)
    - All others  (x=0, x=1, y=0): u = 0, v = 0  (no-slip)

Discretization:
    Collocated finite differences on a uniform n×n structured grid (DMDA).
    Each grid point stores 3 degrees of freedom: (u, v, p).
    The nonlinear system is solved with Newton's method (SNES).
    Pressure is pinned at one corner to remove the hydrostatic null space.

Reference Reynolds numbers:
    Re = 100  → single steady vortex, easy to converge
    Re = 400  → vortex shifts, secondary corner eddies appear
    Re = 1000 → requires finer grid (n ≥ 64) for accuracy

Usage:
    python tutorial_module.py              # default: n=32, Re=100
    python tutorial_module.py -Re 400     # via PETSc options
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

# ---------------------------------------------------------------------------
# Degree-of-freedom layout at each grid node (i, j):
#   dof 0 → u  (x-velocity)
#   dof 1 → v  (y-velocity)
#   dof 2 → p  (pressure)
# ---------------------------------------------------------------------------
U_DOF = 0
V_DOF = 1
P_DOF = 2


class CavitySolver:
    """
    Solves the steady 2D incompressible Navier-Stokes equations for
    lid-driven cavity flow using PETSc DMDA + SNES.

    The momentum equations are discretized with central differences for
    both the diffusive (1/Re)∇²u term and the convective (u·∇)u term.
    The pressure gradient and divergence-free constraint use central
    differences. No turbulence model is applied; this is a direct
    numerical approach valid for moderate Reynolds numbers.
    """

    def __init__(self, n: int = 32, Re: float = 100.0):
        """
        Initialize the solver and build the PETSc distributed array.

        Parameters
        ----------
        n : int
            Number of grid points along each axis (grid is n×n).
            Minimum recommended: 32. Use 64+ for Re > 400.
        Re : float
            Reynolds number. Re = U_lid * L / ν where U_lid = L = 1
            in non-dimensional form and ν is kinematic viscosity.
            Controls the ratio of inertial to viscous forces.
        """
        self.n = n
        self.Re = Re
        self.h = 1.0 / (n - 1)   # uniform grid spacing in x and y

        # Create a 2-D structured grid with 3 DOFs per node.
        # stencil_width=1 gives us access to immediate neighbors
        # (needed for finite-difference stencils).
        self.da = PETSc.DMDA().create(
            dim=2,
            sizes=[n, n],
            dof=3,
            stencil_width=1,
            stencil_type=PETSc.DMDA.StencilType.STAR,
            boundary_type=(PETSc.DM.BoundaryType.GHOSTED,
                           PETSc.DM.BoundaryType.GHOSTED),
            comm=PETSc.COMM_WORLD,
        )
        self.da.setUniformCoordinates(xmin=0.0, xmax=1.0,
                                      ymin=0.0, ymax=1.0)

    # ------------------------------------------------------------------
    # Residual (right-hand side of F(x) = 0)
    # ------------------------------------------------------------------

    def formFunction(self, snes, X: PETSc.Vec, F: PETSc.Vec) -> None:
        """
        Evaluate the nonlinear residual vector F(X) for the coupled
        momentum + continuity system.

        For each interior node (i, j) the residual has three components:

        Momentum-x:
            F_u = u*(u_E - u_W)/(2h) + v*(u_N - u_S)/(2h)
                  - (1/Re)*(u_E - 2u + u_W + u_N - 2u + u_S)/h²
                  + (p_E - p_W)/(2h)

        Momentum-y (analogous with v):
            F_v = u*(v_E - v_W)/(2h) + v*(v_N - v_S)/(2h)
                  - (1/Re)*(v_E - 2v + v_W + v_N - 2v + v_S)/h²
                  + (p_N - p_S)/(2h)

        Continuity:
            F_p = (u_E - u_W)/(2h) + (v_N - v_S)/(2h)

        Boundary and pressure-pin nodes use Dirichlet residuals:
            F = x_local - x_prescribed

        Parameters
        ----------
        snes : PETSc.SNES
            The nonlinear solver context (passed automatically by PETSc).
        X : PETSc.Vec
            Current global solution vector (u, v, p at every node).
        F : PETSc.Vec
            Output residual vector to be populated by this function.
        """
        n = self.n
        h = self.h
        Re = self.Re

        # Scatter global → local (fills ghost points from neighbors)
        localX = self.da.getLocalVec()
        self.da.globalToLocal(X, localX)

        # View as 3D numpy array: shape (nx_local, ny_local, 3)
        x = self.da.getVecArray(localX)
        f = self.da.getVecArray(F)

        # Ranges of nodes owned by this MPI rank (no ghosts)
        (xs, xe), (ys, ye) = self.da.getRanges()

        for j in range(ys, ye):
            for i in range(xs, xe):

                # ---- Pressure pin: fix p=0 at bottom-left corner ----
                if i == 0 and j == 0:
                    f[i, j, P_DOF] = x[i, j, P_DOF]        # p = 0
                    f[i, j, U_DOF] = x[i, j, U_DOF]        # u = 0
                    f[i, j, V_DOF] = x[i, j, V_DOF]        # v = 0
                    continue

                # ---- Boundary nodes: no-slip / moving lid ----
                on_boundary = (i == 0 or i == n-1 or
                               j == 0 or j == n-1)
                if on_boundary:
                    # u-velocity BC
                    u_bc = 1.0 if j == n-1 else 0.0   # lid moves right
                    f[i, j, U_DOF] = x[i, j, U_DOF] - u_bc
                    f[i, j, V_DOF] = x[i, j, V_DOF] - 0.0
                    # Pressure: zero normal gradient (Neumann) via 1st-order
                    if i == 0:
                        f[i, j, P_DOF] = x[i, j, P_DOF] - x[i+1, j, P_DOF]
                    elif i == n-1:
                        f[i, j, P_DOF] = x[i, j, P_DOF] - x[i-1, j, P_DOF]
                    elif j == 0:
                        f[i, j, P_DOF] = x[i, j, P_DOF] - x[i, j+1, P_DOF]
                    else:  # j == n-1 (top)
                        f[i, j, P_DOF] = x[i, j, P_DOF] - x[i, j-1, P_DOF]
                    continue

                # ---- Interior nodes: momentum + continuity ----
                u  = x[i,   j,   U_DOF]
                v  = x[i,   j,   V_DOF]
                uE = x[i+1, j,   U_DOF]
                uW = x[i-1, j,   U_DOF]
                uN = x[i,   j+1, U_DOF]
                uS = x[i,   j-1, U_DOF]
                vE = x[i+1, j,   V_DOF]
                vW = x[i-1, j,   V_DOF]
                vN = x[i,   j+1, V_DOF]
                vS = x[i,   j-1, V_DOF]
                pE = x[i+1, j,   P_DOF]
                pW = x[i-1, j,   P_DOF]
                pN = x[i,   j+1, P_DOF]
                pS = x[i,   j-1, P_DOF]

                inv_h  = 1.0 / h
                inv_h2 = 1.0 / (h * h)

                # Convection (central differences)
                conv_u = u * (uE - uW) * 0.5 * inv_h \
                       + v * (uN - uS) * 0.5 * inv_h
                conv_v = u * (vE - vW) * 0.5 * inv_h \
                       + v * (vN - vS) * 0.5 * inv_h

                # Diffusion: (1/Re) * Laplacian
                diff_u = (1.0/Re) * (uE - 2*u + uW + uN - 2*u + uS) * inv_h2
                diff_v = (1.0/Re) * (vE - 2*v + vW + vN - 2*v + vS) * inv_h2

                # Pressure gradient (central differences)
                dp_dx = (pE - pW) * 0.5 * inv_h
                dp_dy = (pN - pS) * 0.5 * inv_h

                # Divergence (continuity)
                div = (uE - uW) * 0.5 * inv_h + (vN - vS) * 0.5 * inv_h

                # Residuals
                f[i, j, U_DOF] = conv_u - diff_u + dp_dx
                f[i, j, V_DOF] = conv_v - diff_v + dp_dy
                f[i, j, P_DOF] = div

        self.da.restoreLocalVec(localX)

    # ------------------------------------------------------------------
    # Solver setup and execution
    # ------------------------------------------------------------------

    def solve(self) -> PETSc.Vec:
        """
        Configure the PETSc SNES nonlinear solver and compute the
        steady-state solution.

        Uses finite-difference coloring to approximate the Jacobian
        (snes_fd_color option), which avoids the need to hand-code the
        analytic Jacobian. The linear system inside each Newton step is
        solved with the default KSP (GMRES + ILU preconditioner).

        Returns
        -------
        PETSc.Vec
            Converged global solution vector. Values are ordered as
            (u, v, p) at each grid node according to the DMDA layout.
        """
        snes = PETSc.SNES().create(comm=PETSc.COMM_WORLD)

        # Allocate solution and residual vectors
        x = self.da.createGlobalVec()   # solution: starts at zero (good IC)
        f = self.da.createGlobalVec()   # residual

        snes.setFunction(self.formFunction, f)

        # Jacobian via finite-difference coloring (graph-coloring reduces
        # the number of function evaluations needed to fill J)
        J = self.da.createMatrix()
        snes.setJacobian(None, J)

        # Convergence tolerances
        snes.setTolerances(
            rtol=1e-8,    # relative decrease in residual norm
            atol=1e-10,   # absolute residual norm
            stol=1e-8,    # step length tolerance
            max_it=50,    # max Newton iterations
        )

        # Allow command-line overrides (e.g. -snes_monitor)
        snes.setFromOptions()

        # Solve F(x) = 0
        snes.solve(None, x)

        reason = snes.getConvergedReason()
        iters  = snes.getIterationNumber()
        print(f"\n{'='*50}")
        print(f"  SNES converged reason : {reason}")
        print(f"  Newton iterations     : {iters}")
        print(f"  Re = {self.Re},  n = {self.n}x{self.n}")
        print(f"{'='*50}\n")

        return x

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_solution(self, x: PETSc.Vec) -> None:
        """
        Produce a two-panel figure showing:
          1. Filled contour map of the u-velocity (horizontal component).
          2. Streamline plot of the full velocity field (u, v).

        Parameters
        ----------
        x : PETSc.Vec
            Converged solution vector returned by ``solve()``.
        """
        arr = self.da.getVecArray(x)        # shape: (nx, ny, 3)
        u_field = arr[:, :, U_DOF].T        # transpose → (ny, nx) for imshow
        v_field = arr[:, :, V_DOF].T
        p_field = arr[:, :, P_DOF].T

        n = self.n
        xi = np.linspace(0, 1, n)
        yi = np.linspace(0, 1, n)
        X, Y = np.meshgrid(xi, yi)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Lid-Driven Cavity Flow  |  Re = {self.Re}  |  "
            f"Grid = {n}×{n}",
            fontsize=14, fontweight='bold'
        )

        # --- Panel 1: u-velocity contour ---
        ax = axes[0]
        cf = ax.contourf(X, Y, u_field, levels=30, cmap='RdBu_r')
        fig.colorbar(cf, ax=ax, label='u-velocity')
        ax.set_title('u-velocity  (horizontal)')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')

        # --- Panel 2: streamlines ---
        ax = axes[1]
        speed = np.sqrt(u_field**2 + v_field**2)
        strm = ax.streamplot(
            X, Y, u_field, v_field,
            color=speed, cmap='plasma',
            linewidth=1.2, density=1.8,
            arrowsize=1.2,
        )
        fig.colorbar(strm.lines, ax=ax, label='Speed |u|')
        ax.set_title('Streamlines')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig('cavity_flow_solution.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure saved to cavity_flow_solution.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Read Re from PETSc options if provided: -Re 400
    opts = PETSc.Options()
    Re = opts.getReal('Re', default=100.0)
    n  = opts.getInt('n',  default=32)

    solver   = CavitySolver(n=n, Re=Re)
    solution = solver.solve()
    solver.plot_solution(solution)