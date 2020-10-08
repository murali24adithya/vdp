class PartialInterpretationException(Exception):
    """
    This exception is raised when a model is queried for an interpretation that is absent. One way of ensuring this
    does not happen is to 'close' model interpretations with default values for each function (false for predicates).
    """
    pass
