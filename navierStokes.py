import sys
import numpy as np
import matplotlib.pyplot as plt
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

U_DOF = 0
V_DOF = 1
P_DOF = 2

class CavitySolver:
    def __init__(self, n=32, Re=100.0):
        self.n = n
        self.Re = Re
        self.h = 1.0 / (n - 1)
        self.da = PETSc.DMDA().create(
            dim=2, sizes=[n, n], dof=3, stencil_width=1,
            stencil_type=PETSc.DMDA.StencilType.STAR,
            boundary_type=(PETSc.DM.BoundaryType.NONE, PETSc.DM.BoundaryType.NONE),
            comm=PETSc.COMM_WORLD,
        )
        self.da.setUniformCoordinates(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

    def applyIC(self, x):
        n = self.n
        arr = x.getArray().copy().reshape(n, n, 3)
        arr[:, n-1, U_DOF] = 1.0
        x.setArray(arr.ravel())

    def residual(self, X):
        """Pure numpy residual — returns flat array."""
        n, h, Re = self.n, self.h, self.Re
        x = np.array(X).reshape(n, n, 3)
        f = np.zeros((n, n, 3))
        for j in range(n):
            for i in range(n):
                if i == 0 and j == 0:
                    f[i,j,P_DOF] = x[i,j,P_DOF]
                    f[i,j,U_DOF] = x[i,j,U_DOF]
                    f[i,j,V_DOF] = x[i,j,V_DOF]
                    continue
                on_boundary = (i==0 or i==n-1 or j==0 or j==n-1)
                if on_boundary:
                    u_bc = 1.0 if j == n-1 else 0.0
                    f[i,j,U_DOF] = x[i,j,U_DOF] - u_bc
                    f[i,j,V_DOF] = x[i,j,V_DOF]
                    if i == 0:
                        f[i,j,P_DOF] = x[i,j,P_DOF] - x[i+1,j,P_DOF]
                    elif i == n-1:
                        f[i,j,P_DOF] = x[i,j,P_DOF] - x[i-1,j,P_DOF]
                    elif j == 0:
                        f[i,j,P_DOF] = x[i,j,P_DOF] - x[i,j+1,P_DOF]
                    else:
                        f[i,j,P_DOF] = x[i,j,P_DOF] - x[i,j-1,P_DOF]
                    continue
                u  = x[i,j,U_DOF];   v  = x[i,j,V_DOF]
                uE = x[i+1,j,U_DOF]; uW = x[i-1,j,U_DOF]
                uN = x[i,j+1,U_DOF]; uS = x[i,j-1,U_DOF]
                vE = x[i+1,j,V_DOF]; vW = x[i-1,j,V_DOF]
                vN = x[i,j+1,V_DOF]; vS = x[i,j-1,V_DOF]
                pE = x[i+1,j,P_DOF]; pW = x[i-1,j,P_DOF]
                pN = x[i,j+1,P_DOF]; pS = x[i,j-1,P_DOF]
                inv_h  = 1.0 / h
                inv_h2 = 1.0 / (h * h)
                conv_u = u*(uE-uW)*0.5*inv_h + v*(uN-uS)*0.5*inv_h
                conv_v = u*(vE-vW)*0.5*inv_h + v*(vN-vS)*0.5*inv_h
                diff_u = (1.0/Re)*(uE-2*u+uW+uN-2*u+uS)*inv_h2
                diff_v = (1.0/Re)*(vE-2*v+vW+vN-2*v+vS)*inv_h2
                dp_dx  = (pE-pW)*0.5*inv_h
                dp_dy  = (pN-pS)*0.5*inv_h
                div    = (uE-uW)*0.5*inv_h + (vN-vS)*0.5*inv_h
                f[i,j,U_DOF] = conv_u - diff_u + dp_dx
                f[i,j,V_DOF] = conv_v - diff_v + dp_dy
                f[i,j,P_DOF] = div
        return f.ravel()

    def formFunction(self, snes, X, F):
        f = self.residual(X.copy().getArray())
        F.setArray(f)

    def formJacobian(self, snes, X, J, P):
        """Finite-difference Jacobian via forward differences."""
        x0 = X.copy().getArray()
        f0 = self.residual(x0)
        N  = len(x0)
        eps = 1e-6
        P.zeroEntries()
        for k in range(N):
            xp = x0.copy()
            xp[k] += eps
            fp = self.residual(xp)
            col = (fp - f0) / eps
            for row in range(N):
                if col[row] != 0.0:
                    P.setValue(row, k, col[row])
        P.assemblyBegin()
        P.assemblyEnd()
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    def solve(self):
        opts = PETSc.Options()
        opts.setValue('ksp_type', 'preonly')
        opts.setValue('pc_type', 'lu')

        snes = PETSc.SNES().create(comm=PETSc.COMM_WORLD)
        x = self.da.createGlobalVec()
        f = self.da.createGlobalVec()
        self.applyIC(x)
        snes.setFunction(self.formFunction, f)

        J = self.da.createMatrix()
        J.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        snes.setJacobian(self.formJacobian, J)

        snes.setTolerances(rtol=1e-6, atol=1e-8, stol=1e-8, max_it=50)
        snes.setFromOptions()
        snes.solve(None, x)

        reason = snes.getConvergedReason()
        iters  = snes.getIterationNumber()
        print(f"\n{'='*50}")
        print(f"  SNES converged reason : {reason}")
        print(f"  Newton iterations     : {iters}")
        print(f"  Re = {self.Re},  n = {self.n}x{self.n}")
        print(f"{'='*50}\n")
        return x

    def plot_solution(self, x):
        n = self.n
        arr = x.getArray().copy().reshape(n, n, 3)
        u_field = arr[:,:,U_DOF].T
        v_field = arr[:,:,V_DOF].T
        xi = np.linspace(0, 1, n)
        yi = np.linspace(0, 1, n)
        X, Y = np.meshgrid(xi, yi)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Lid-Driven Cavity Flow  |  Re = {self.Re}  |  Grid = {n}x{n}",
                     fontsize=14, fontweight='bold')
        ax = axes[0]
        cf = ax.contourf(X, Y, u_field, levels=30, cmap='RdBu_r')
        fig.colorbar(cf, ax=ax, label='u-velocity')
        ax.set_title('u-velocity (horizontal)')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax = axes[1]
        speed = np.sqrt(u_field**2 + v_field**2)
        strm = ax.streamplot(xi, yi, u_field, v_field,
                             color=speed, cmap='plasma',
                             linewidth=1.2, density=1.8, arrowsize=1.2)
        fig.colorbar(strm.lines, ax=ax, label='Speed |u|')
        ax.set_title('Streamlines')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        plt.tight_layout()
        plt.savefig('cavity_flow_solution.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure saved to cavity_flow_solution.png")

if __name__ == '__main__':
    opts = PETSc.Options()
    Re = opts.getReal('Re', default=100.0)
    n  = opts.getInt('n',  default=32)
    solver   = CavitySolver(n=n, Re=Re)
    solution = solver.solve()
    solver.plot_solution(solution)
