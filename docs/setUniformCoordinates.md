# `DMDA.setUniformCoordinates`

**PETSc GitLab Source:**
- Header: [`include/petscdmda.h`](https://gitlab.com/petsc/petsc/-/blob/main/include/petscdmda.h)
- Implementation: [`src/dm/impls/da/dalocal.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/dm/impls/da/dalocal.c)
- Key lines: [`DMDASetUniformCoordinates` in dalocal.c, line 318](https://gitlab.com/petsc/petsc/-/blob/main/src/dm/impls/da/dalocal.c#L318)

---

## Cython Bridge

In the petsc4py source (`src/binding/petsc4py/src/petsc4py/PETSc/DMDA.pyx`), the
Python method resolves to:

```cython
def setUniformCoordinates(
    self,
    xmin=0, xmax=1,
    ymin=0, ymax=1,
    zmin=0, zmax=1,
):
    cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
    cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
    cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
    CHKERR( DMDASetUniformCoordinates(
        self.dm,
        _xmin, _xmax,
        _ymin, _ymax,
        _zmin, _zmax,
    ) )
```

The underlying C routine called is **`DMDASetUniformCoordinates`** in
`src/dm/impls/da/dalocal.c`.

---

## Docstring

```python
def setUniformCoordinates(
    self,
    xmin: float = 0.0,
    xmax: float = 1.0,
    ymin: float = 0.0,
    ymax: float = 1.0,
    zmin: float = 0.0,
    zmax: float = 1.0,
) -> None:
    """
    Set uniformly-spaced coordinates on the DMDA's physical domain.

    Maps each logical grid index to a physical coordinate by dividing
    the specified bounding box into equally-spaced intervals. For a
    1D DMDA with ``n`` global grid points and bounds ``[xmin, xmax]``,
    the coordinate of node ``i`` is:

    Mathematics
    ----------
        x_i = x_{\\min} + i \\cdot h, \\quad
        h = \\frac{x_{\\max} - x_{\\min}}{n - 1}, \\quad
        i = 0, 1, \\ldots, n-1.

    For a 2D DMDA with ``(nx, ny)`` points the same formula is applied
    independently in each direction, producing a Cartesian product mesh.
    The coordinate vector is stored internally in the DM and can be
    retrieved with :py:meth:`getCoordinates`.

    This function takes the grid, which PETSc thinks of as just a table of numbered rows and columns, and tells it where those grid points actually live in physical space. Give it the corners of thedomain (for the cavity solver, the unit square from 0 to 1 in both directions), and it evenly spaces all the grid points in between. For a 32×32 grid on [0,1], that means each neighbouring pair of points is exactly 1/31 apart. This spacing value, called h, is what the finite-difference stencil divides by when it approximates derivatives.

    
    Parameters
    ----------
    xmin : float
        Lower bound of the physical domain in the x-direction.
        Corresponds to ``PetscReal xmin`` in C.
    xmax : float
        Upper bound of the physical domain in the x-direction.
        Corresponds to ``PetscReal xmax`` in C.
    ymin : float, optional
        Lower bound in the y-direction. Required for 2-D and 3-D DMDAs.
        Ignored for 1-D DMDAs. Default is ``0.0``.
    ymax : float, optional
        Upper bound in the y-direction. Required for 2-D and 3-D DMDAs.
        Default is ``1.0``.
    zmin : float, optional
        Lower bound in the z-direction. Required for 3-D DMDAs only.
        Default is ``0.0``.
    zmax : float, optional
        Upper bound in the z-direction. Required for 3-D DMDAs only.
        Default is ``1.0``.

    Returns
    -------
    None

    Raises
    ------
    petsc4py.PETSc.Error
        If the DMDA has not been set up, or if ``xmin >= xmax``
        (and analogously for y and z), PETSc raises a non-zero
        error code which petsc4py converts to this exception.

    See Also
    --------
    getCoordinates : Retrieve the coordinate vector set by this method.
    setUniformCoordinatesExplicit : Set coordinates from a user-supplied array.
    getRanges : Obtain the local index ranges owned by this MPI rank.

    Minimum Working Example (MWE)
    --------
    **2-D unit-square domain (lid-driven cavity):**

    >>> import petsc4py, sys
    >>> petsc4py.init(sys.argv)
    >>> from petsc4py import PETSc
    >>> n = 32
    >>> da = PETSc.DMDA().create(dim=2, sizes=[n, n], dof=3,
    ...                          stencil_width=1)
    >>> da.setUniformCoordinates(xmin=0.0, xmax=1.0,
    ...                          ymin=0.0, ymax=1.0)
    >>> coords = da.getCoordinates()   # PETSc.Vec of shape (n*n*2,)

    **1-D domain on [0, 2π]:**

    >>> import math
    >>> da1d = PETSc.DMDA().create(dim=1, sizes=[64])
    >>> da1d.setUniformCoordinates(xmin=0.0, xmax=2*math.pi)
    """
```
---

## Parameter Translation Table

| C type | C name | Python type | Python name | Notes |
|--------|--------|-------------|-------------|-------|
| `PetscReal` | `xmin` | `float` | `xmin` | lower x bound |
| `PetscReal` | `xmax` | `float` | `xmax` | upper x bound |
| `PetscReal` | `ymin` | `float` | `ymin` | lower y bound (2D/3D) |
| `PetscReal` | `ymax` | `float` | `ymax` | upper y bound (2D/3D) |
| `PetscReal` | `zmin` | `float` | `zmin` | lower z bound (3D only) |
| `PetscReal` | `zmax` | `float` | `zmax` | upper z bound (3D only) |
| `DM` | `da` (self) | `PETSc.DMDA` | `self` | structured grid object |
| `PetscErrorCode` | return | `None` / raises | — | error via `CHKERR` |
