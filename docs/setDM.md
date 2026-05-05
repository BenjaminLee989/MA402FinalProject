# `SNES.setDM`

**PETSc GitLab Source:**
- Header: [`include/petscsnes.h`](https://gitlab.com/petsc/petsc/-/blob/main/include/petscsnes.h)
- Implementation: [`src/snes/interface/snes.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/snes/interface/snes.c)
- Key lines: [`SNESSetDM` in snes.c, line 3394](https://gitlab.com/petsc/petsc/-/blob/main/src/snes/interface/snes.c#L3394)

---

## Cython Bridge

In `src/binding/petsc4py/src/petsc4py/PETSc/SNES.pyx` the wrapper is:

```cython
def setDM(self, DM dm):
    CHKERR( SNESSetDM(self.snes, dm.dm) )
```

The underlying C routine called is **`SNESSetDM`** in
`src/snes/interface/snes.c`. Internally, `SNESSetDM` stores a reference
to the DM on the SNES object, and if finite-difference coloring is
requested via `-snes_fd_color`, automatically invokes
`DMSNESSetFunctionLocal` and sets up the coloring context by calling
`DMCreateColoring` on the stored DM before the first Newton step.

---

## Docstring

```python
def setDM(self, dm: PETSc.DM) -> None:
    """
    Attach a DM (domain manager) to the SNES nonlinear solver.

    Associates a :py:class:`PETSc.DM` with this SNES context. Once
    attached, PETSc uses the DM to:

    1. **Construct the coloring** for matrix-free finite-difference
       Jacobian approximation when ``-snes_fd_color`` is set. The DM's
       stencil graph is used by :c:func:`DMCreateColoring` to assign
       colors to grid nodes such that no two same-colored nodes share a
       stencil coupling, allowing the full Jacobian columns to be
       approximated with ``n_colors`` residual evaluations instead of
       ``n_dof`` evaluations.

    2. **Propagate mesh information** (local-to-global mappings, ghost
       point layouts) to linear sub-solvers (KSP/PC) created internally
       by SNES, so they can apply DM-aware preconditioners.

    3. **Enable ``setFromOptions`` hooks** that read DM-specific
       command-line options (e.g. ``-snes_grid_sequence``).

    Without this call, passing ``None`` as the matrix argument to
    :py:meth:`setJacobian` and relying on ``-snes_fd_color`` will
    raise a PETSc error at solve time because no coloring context exists.

    Mathematics
    ---------
    Given a Jacobian $J \in \mathbb{R}^{n \times n}$, finite-difference coloring approximates:

    $$J_{ij} \approx \frac{F_i(x + h e_j) - F_i(x)}{h}$$

    With $c$ colors, the total residual evaluations reduces from $n$ to $c$, where $c \ll n$ for sparse stencils.
Newton's method needs a Jacobian matrix at every iteration, but computing one entry at a time for a 3000×3000 matrix would be prohibitively slow. This function solves that problem by giving the solver access to the grid's connectivity information, which it uses to compute the entire Jacobian with only 15 residual evaluations instead of 3072. The solution is called finite-difference coloring: nodes that don't influence each other through the stencil get assigned the same color, and all same-colored nodes can be perturbed simultaneously in a single residual evaluation. Without calling setDM first, the solver has no way to compute this coloring, and the -snes_fd_color option you set at the top of the script will fail at runtime.

    Parameters
    ----------
    dm : PETSc.DM
        The domain manager to attach. Typically a :py:class:`PETSc.DMDA`
        for structured grids or a :py:class:`PETSc.DMPlex` for
        unstructured meshes. The DM must already be fully set up
        (i.e. :py:meth:`PETSc.DMDA.create` or equivalent has been called).
        Corresponds to ``DM dm`` (``PetscObject``) in C.

    Returns
    -------
    None

    Raises
    ------
    petsc4py.PETSc.Error
        If ``dm`` has not been set up, or if the SNES object is in an
        incompatible state, PETSc raises a non-zero error code which
        petsc4py converts to this exception.

    See Also
    --------
    PETSc.SNES.getDM : Retrieve the DM currently attached to this SNES.
    PETSc.SNES.setJacobian : Supply the Jacobian matrix structure;
        ``setDM`` must precede this call when using finite-difference
        coloring.
    PETSc.SNES.setFromOptions : Reads command-line options that may depend
        on the attached DM (e.g. ``-snes_fd_color``).
    PETSc.DMDA.createMatrix : Creates the preallocated Jacobian matrix
        whose sparsity structure drives the finite-difference coloring.

    Minimum Working Example (MWE)
    --------
    **Attaching a DMDA to SNES for finite-difference coloring (cavity
    solver pattern):**

    >>> import petsc4py, sys
    >>> petsc4py.init(sys.argv)
    >>> from petsc4py import PETSc
    >>> n = 32
    >>> da = PETSc.DMDA().create(
    ...     dim=2, sizes=[n, n], dof=3,
    ...     stencil_width=1,
    ...     stencil_type=PETSc.DMDA.StencilType.STAR,
    ... )
    >>> snes = PETSc.SNES().create()
    >>> f = da.createGlobalVec()
    >>> snes.setFunction(residual_fn, f)
    >>> snes.setDM(da)                   # attach DM before setJacobian
    >>> J = da.createMatrix()
    >>> snes.setJacobian(None, J)        # None → FD coloring uses DM
    >>> snes.setFromOptions()
    >>> x = da.createGlobalVec()
    >>> snes.solve(None, x)

    **Verifying the attached DM after the call:**

    >>> retrieved_dm = snes.getDM()
    >>> assert retrieved_dm == da
    """
```

---

## Parameter Translation Table

| C type | C name | Python type | Python name | Notes |
|--------|--------|-------------|-------------|-------|
| `SNES` | `snes` (self) | `PETSc.SNES` | `self` | nonlinear solver |
| `DM` | `dm` | `PETSc.DM` | `dm` | must be fully set-up |
| `PetscErrorCode` | return | `None` / raises | — | error via `CHKERR` |
