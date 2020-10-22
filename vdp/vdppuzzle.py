class VDPPuzzle:
    """
    This class represents a single VDP Puzzle. It contains the vocabulary of the problem,
    and a list of training and candidate model objects. Each model object is an instance of the FOModel class.
    """

    def __init__(self, vocabulary=None, training_models=None, candidate_models=None):
        self.vocabulary = vocabulary
        self.training_models = training_models
        self.candidate_models = candidate_models
        self.puzzle_name = ""

    # Setter methods for puzzle attributes
    def _set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary

    def _set_training_models(self, training_models):
        self.training_models = training_models

    def _set_candidate_models(self, candidate_models):
        self.candidate_models = candidate_models

    # Getter methods for puzzle attributes
    def get_vocabulary(self):
        return self.vocabulary

    def get_training_models(self):
        return self.training_models

    def get_candidate_models(self):
        return self.candidate_models
