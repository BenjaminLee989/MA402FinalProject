# `DMDA.createMatrix`

**PETSc GitLab Source:**
- Header: [`include/petscdm.h`](https://gitlab.com/petsc/petsc/-/blob/main/include/petscdm.h)
- Implementation: [`src/dm/impls/da/da.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/dm/impls/da/da.c)
  and [`src/dm/interface/dm.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/dm/interface/dm.c)
- Key lines: search [`DMCreateMatrix` in dm.c, line 857](https://gitlab.com/petsc/petsc/-/blob/main/src/dm/interface/dm.c#L857)

---

## Cython Bridge

In `src/binding/petsc4py/src/petsc4py/PETSc/DM.pyx` the wrapper is:

```cython
def createMatrix(self):
    cdef Mat mat = Mat()
    CHKERR( DMCreateMatrix(self.dm, &mat.mat) )
    PetscINCREF(mat.obj)
    return mat
```

The underlying C routine is **`DMCreateMatrix`**. For a DMDA object this
dispatches to the DA-specific implementation which reads the stencil type,
stencil width, and degrees-of-freedom (dof) that were set at
:py:meth:`create` time, then calls `MatMPIAIJSetPreallocation` (or the
sequential equivalent) automatically.

---

## Docstring

```python
def createMatrix(self) -> PETSc.Mat:
    """
    Allocate a sparse matrix whose non-zero pattern matches the DMDA stencil.

    Constructs a parallel sparse matrix (type ``MATAIJ`` by default) whose
    sparsity structure is derived automatically from the DMDA's dimension,
    stencil type, stencil width, and degrees of freedom per node. The
    resulting matrix is suitable for use as a Jacobian passed to
    :py:meth:`PETSc.SNES.setJacobian`.

    **Sparsity pattern derivation.** For a 2-D DMDA with ``dof = d``,
    ``stencil_type = STAR``, and ``stencil_width = w``, each block-row
    (corresponding to one grid node) couples to at most

    Mathematics
    --------

        N_{\\text{nz}} = (2w + 1)^2 - 4\\binom{w}{2} \\; \\text{(STAR)}
        \\quad \\text{or} \\quad
        N_{\\text{nz}} = (2w + 1)^2 \\; \\text{(BOX)}

    neighbouring nodes. The actual number of matrix non-zeros per row is
    ``d * N_nz * d`` (coupling all dof at each neighbouring node to all
    dof at the current node). For the lid-driven cavity with ``dof=3``,
    ``STAR`` stencil, ``stencil_width=1``, each row has at most
    ``3 * 5 * 3 = 45`` non-zeros.

    No numerical values are stored; call this method to obtain the correct
    *structure*, then fill entries during Jacobian assembly.


    When PETSc's Newton solver needs to compute how the residual changes as the solution changes, it needs a matrix — the Jacobian — to store those relationships. But before it can fill in any numbers, it needs to know the shape of that matrix: specifically, which entries are allowed to be non-zero. This function builds that shape automatically by looking at the grid's stencil. For the cavity solver, each grid node talks to its 4 immediate neighbours (up, down, left, right) and itself, and each node stores 3 values (u, v, p), so each row of the matrix can have at most 45 non-zero entries. Preallocating this pattern upfront is what makes the solver memory-efficient — PETSc reserves exactly the space it needs and no more, rather than discovering the pattern piece by piece during assembly.

    Parameters
    ----------
    None

    Returns
    -------
    PETSc.Mat
        A preallocated sparse matrix (``PetscMat``) with:

        - global size ``(n_total * dof) × (n_total * dof)`` where
          ``n_total`` is the total number of grid nodes,
        - local ownership rows consistent with this rank's DMDA partition,
        - ``MAT_NEW_NONZERO_ALLOCATION_ERR`` set to ``True`` (PETSc will
          error if you insert an entry outside the preallocated pattern).

    Raises
    ------
    petsc4py.PETSc.Error
        If the DM has not been set up (``DMSetUp`` not called), or if
        memory allocation fails, PETSc raises a non-zero error code which
        petsc4py converts to this exception.

    See Also
    --------
    PETSc.SNES.setJacobian : Pass the matrix returned here as the
        preconditioning matrix ``P``.
    PETSc.DMDA.create : The stencil type and width set here determine the
        sparsity pattern computed by ``createMatrix``.
    PETSc.Mat.setValuesStencil : Fill the preallocated matrix with
        Jacobian entries using DMDA stencil indices.

    Minimum Working Example (MWE)
    --------
    **Create and inspect the Jacobian matrix for the cavity solver:**

    >>> import petsc4py, sys
    >>> petsc4py.init(sys.argv)
    >>> from petsc4py import PETSc
    >>> n = 32
    >>> da = PETSc.DMDA().create(
    ...     dim=2, sizes=[n, n], dof=3,
    ...     stencil_width=1,
    ...     stencil_type=PETSc.DMDA.StencilType.STAR,
    ... )
    >>> J = da.createMatrix()
    >>> print(J.getSize())     # global (rows, cols)
    (3072, 3072)               # n*n*dof = 32*32*3
    >>> print(J.getType())
    'aij'

    **Pass to SNES as the Jacobian structure:**

    >>> snes = PETSc.SNES().create()
    >>> snes.setJacobian(None, J)   # None = use finite-difference coloring
    """
```

---

## Parameter Translation Table

| C type | C name | Python type | Python name | Notes |
|--------|--------|-------------|-------------|-------|
| `DM` | `da` (self) | `PETSc.DMDA` | `self` | must be set-up |
| `Mat *` | `J` (output) | `PETSc.Mat` | return value | caller owns ref |
| `PetscErrorCode` | return | `None` / raises | — | error via `CHKERR` |
