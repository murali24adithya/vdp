class PartialInterpretationException(Exception):
    """
    This exception is raised when a model is queried for an interpretation that is absent. One way of ensuring this
    does not happen is to 'close' model interpretations with default values for each function (false for predicates).
    """
    pass


class MalformedPuzzleException(Exception):
    """
    This exception is raised when a VDPPuzzle object does not fit the assumptions made by a solver.
    Example: empty set of training/candidate images.
    """
    pass


class NonsenseSolverConfigurationException(Exception):
    """
    This exception is raised when a solver's configuration does not make sense with respect to its assumptions.
    Example: negative number of quantified variables.
    """
    pass
