class VariationalPosterior:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Variational inference classes are outside the ssm_new_full GLM-HMM subset."
        )


class SLDSStructuredMeanFieldVariationalPosterior(VariationalPosterior):
    pass


class SLDSTriDiagVariationalPosterior(VariationalPosterior):
    pass
