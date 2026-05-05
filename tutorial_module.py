import sys
import numpy as np
import matplotlib.pyplot as plt

import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

opts = PETSc.Options()
opts["snes_fd_color"]         = None
opts["snes_monitor"]          = None
opts["snes_converged_reason"] = None

U_DOF, V_DOF, P_DOF = 0, 1, 2



class CavitySolver:

    def __init__(self, n=32, Re=100.0):
        self.n  = n
        self.Re = Re
        self.h  = 1.0 / (n - 1)

        self.da = PETSc.DMDA().create(
            dim=2,
            sizes=[n, n],
            dof=3,
            stencil_width=1,
            stencil_type=PETSc.DMDA.StencilType.STAR,
        )
        self.da.setUniformCoordinates(0, 1, 0, 1)

    def formFunction(self, snes, X, F):
        n, h, Re = self.n, self.h, self.Re
        x = X.getArray(readonly=True).reshape(n, n, 3)
        f = np.zeros((n, n, 3), dtype=float)

        ih  = 1.0 / h
        ih2 = ih * ih

        for j in range(n):
            for i in range(n):

                # Pressure pin at bottom-left corner
                if i == 0 and j == 0:
                    f[j, i, P_DOF] = x[j, i, P_DOF]
                    f[j, i, U_DOF] = x[j, i, U_DOF]
                    f[j, i, V_DOF] = x[j, i, V_DOF]
                    continue

                # Boundary conditions
                if i == 0 or i == n-1 or j == 0 or j == n-1:
                    u_bc = 1.0 if j == n-1 else 0.0
                    f[j, i, U_DOF] = x[j, i, U_DOF] - u_bc
                    f[j, i, V_DOF] = x[j, i, V_DOF]
                    # Neumann pressure: dp/dn = 0
                    if   i == 0:   f[j, i, P_DOF] = x[j, i, P_DOF] - x[j, i+1, P_DOF]
                    elif i == n-1: f[j, i, P_DOF] = x[j, i, P_DOF] - x[j, i-1, P_DOF]
                    elif j == 0:   f[j, i, P_DOF] = x[j, i, P_DOF] - x[j+1, i, P_DOF]
                    else:          f[j, i, P_DOF] = x[j, i, P_DOF] - x[j-1, i, P_DOF]
                    continue

                # Interior nodes
                u, v = x[j, i, U_DOF], x[j, i, V_DOF]

                # Upwind convection (stable for high Re)
                u_up = u*(x[j,i,U_DOF]-x[j,i-1,U_DOF])*ih if u > 0 else u*(x[j,i+1,U_DOF]-x[j,i,U_DOF])*ih
                v_up = v*(x[j,i,U_DOF]-x[j-1,i,U_DOF])*ih if v > 0 else v*(x[j+1,i,U_DOF]-x[j,i,U_DOF])*ih
                conv_u = u_up + v_up

                u_vp = u*(x[j,i,V_DOF]-x[j,i-1,V_DOF])*ih if u > 0 else u*(x[j,i+1,V_DOF]-x[j,i,V_DOF])*ih
                v_vp = v*(x[j,i,V_DOF]-x[j-1,i,V_DOF])*ih if v > 0 else v*(x[j+1,i,V_DOF]-x[j,i,V_DOF])*ih
                conv_v = u_vp + v_vp

                # Central diffusion
                diff_u = (1.0/Re)*(x[j,i+1,U_DOF] + x[j,i-1,U_DOF] + x[j+1,i,U_DOF] + x[j-1,i,U_DOF] - 4*u)*ih2
                diff_v = (1.0/Re)*(x[j,i+1,V_DOF] + x[j,i-1,V_DOF] + x[j+1,i,V_DOF] + x[j-1,i,V_DOF] - 4*v)*ih2

                # Central pressure gradient
                dp_dx = (x[j,i+1,P_DOF] - x[j,i-1,P_DOF])*0.5*ih
                dp_dy = (x[j+1,i,P_DOF] - x[j-1,i,P_DOF])*0.5*ih

                # Continuity + pressure stabilization (prevents checkerboarding)
                div = (x[j,i+1,U_DOF]-x[j,i-1,U_DOF])*0.5*ih + (x[j+1,i,V_DOF]-x[j-1,i,V_DOF])*0.5*ih
                stab = -0.1*h*(x[j,i+1,P_DOF]+x[j,i-1,P_DOF]+x[j+1,i,P_DOF]+x[j-1,i,P_DOF]-4*x[j,i,P_DOF])*ih2

                f[j, i, U_DOF] = conv_u - diff_u + dp_dx
                f[j, i, V_DOF] = conv_v - diff_v + dp_dy
                f[j, i, P_DOF] = div + stab

        F.getArray()[:] = f.ravel()

    def solve(self):
        snes = PETSc.SNES().create()

        x = self.da.createGlobalVec()
        f = self.da.createGlobalVec()

        x.set(0.0)
        arr = x.getArray().reshape(self.n, self.n, 3)
        arr[self.n-1, :, U_DOF] = 1.0
        x.setArray(arr.ravel())

        snes.setFunction(self.formFunction, f)
        snes.setDM(self.da)

        J = self.da.createMatrix()
        snes.setJacobian(None, J)

        snes.setTolerances(rtol=1e-8, atol=1e-10, stol=1e-8, max_it=100)
        snes.setFromOptions()
        snes.solve(None, x)

        reason = snes.getConvergedReason()
        iters  = snes.getIterationNumber()
        print(f"Converged Reason : {reason}")
        print(f"Number of Iterations : {iters}")
        if reason < 0:
            print("WARNING: did not converge.")

        return x

    def plot_solution(self, x):
        n   = self.n
        arr = x.getArray(readonly=True).reshape(n, n, 3)

        u = arr[:, :, U_DOF]
        v = arr[:, :, V_DOF]
        p = arr[:, :, P_DOF]

        speed = np.sqrt(u**2 + v**2)
        xi    = np.linspace(0, 1, n)
        X, Y  = np.meshgrid(xi, xi)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Lid-Driven Cavity Flow  |  Re = {self.Re}  |  Grid = {n}x{n}",
            fontsize=13, fontweight="bold"
        )

        ax = axes[0]
        cf = ax.contourf(X, Y, u, levels=50, cmap="RdBu_r")
        fig.colorbar(cf, ax=ax, label="u-velocity")
        ax.set_title("u-velocity (horizontal)")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax = axes[1]
        cf = ax.contourf(X, Y, v, levels=50, cmap="RdBu_r")
        fig.colorbar(cf, ax=ax, label="v-velocity")
        ax.set_title("v-velocity (vertical)")
        ax.set_xlabel("x")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax = axes[2]
        lw   = 2 * speed / (speed.max() + 1e-12)
        strm = ax.streamplot(X, Y, u, v, color=speed, cmap="plasma",
                             linewidth=lw, density=1.5, arrowsize=1.2)
        fig.colorbar(strm.lines, ax=ax, label="Speed |u|")
        ax.set_title("Streamlines")
        ax.set_xlabel("x")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig("cavity_flow_solution.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Plot saved to cavity_flow_solution.png")


if __name__ == "__main__":
    opts = PETSc.Options()
    Re   = opts.getReal("Re", 100.0)
    n    = opts.getInt("n",   32)

    solver = CavitySolver(n, Re)
    sol    = solver.solve()
    solver.plot_solution(sol)