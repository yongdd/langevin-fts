"""Shared parameter serialization and species-smearing adapters."""


def _drop_none_values(obj):
    """Remove None-valued entries from the parameter copy saved by savemat.

    An absent key has the same meaning as None for these parameters.
    Tuples become lists, matching the historical serialization behavior.
    """
    if isinstance(obj, dict):
        return {k: _drop_none_values(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_drop_none_values(v) for v in obj if v is not None]
    return obj


class _PerSpeciesSmearing:
    """Expose Electrostatics.smear_dict through the mainline Smearing API.

    Density smearing uses the individual species kernels. Model classes
    assemble the propagator inputs, including the electrostatic potential.
    `type` and `params` enter the mainline checkpoint settings, so a
    continuation must use the same charged model.
    """

    def __init__(self, electro, psi_dt_scaling=1.0):
        self._electro = electro
        self.enabled = True
        self.type = "per_species_charged"
        self.params = {"radiuses": dict(electro.radiuses), "charges": dict(electro.charges),
                       "e_coupling": electro.e_coupling, "psi_dt_scaling": float(psi_dt_scaling),
                       "basis": type(electro).__name__}

    def apply_to_dict(self, fields):
        return self._electro.smear_dict(fields)

    def apply(self, field):
        raise NotImplementedError(
            "Per-species smearing has no species-agnostic apply(); "
            "use apply_to_dict.")
