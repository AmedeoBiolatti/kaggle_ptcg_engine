# Oracle Helpers

This package contains Python-side helpers used by parity validation:

- `canonical.py`: canonical state and option projections used for comparisons.
- `random_branch_parity.py`: randomized branch parity checks against the native
  engine when a reference `cg` shared library is available.

The reference shared library is optional and is not included.
