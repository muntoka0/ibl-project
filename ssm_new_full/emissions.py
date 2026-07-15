class Emissions:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Emissions/SLDS classes are outside the ssm_new_full GLM-HMM subset."
        )


class GaussianEmissions(Emissions):
    pass


class GaussianOrthogonalEmissions(Emissions):
    pass
