import VDPExceptions
from VDPVocabulary import *


class VDPFOModel:
    __doc__ = """
    This class represents a single Multi-Sorted First-Order Model. It contains the underlying universe of the model, the 
    vocabulary that the model interprets, and the interpretation. 
    NOTE: There are no checks against partial interpretations, and it is up to the client code 
    to configure the model correctly.
    """

    def __init__(self, elements, vocabulary, interpretation):
        # Elements is a dictionary of symbols denoting the underlying (multi-sorted) universe.
        # The keys are the names of the sorts (which are also the symbol_type of symbols in that sort)
        # The values are a list of elements (symbols) in that sort.
        self.elements = elements
        # Vocabulary is a dictionary of constant/relation/function symbols.
        # The keys are 'Constant', 'Relation'.
        # No function symbols as yet. No distinguishing by signature.
        self.vocabulary = vocabulary
        # Interpretation is a dictionary of dictionaries: relation/function_name -> (argument_tuple -> valuation)
        # For constants the structure is constant_name -> value
        self.interpretation = interpretation

    # Setter methods for model attributes
    def __set_elements(self, elements):
        self.elements = elements

    def __set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary

    def __set_interpretation(self, interpretation):
        self.interpretation = interpretation

    # Getter methods for model attributes
    def get_elements(self):
        return self.elements

    def get_vocabulary(self):
        return self.vocabulary

    # Usage NOT recommended. Use function to query particular the interpretation of a particular term instead.
    def get_interpretation(self):
        return self.interpretation

    # Lookup a single entry in the interpretation.
    def lookup_interpretation(self, function, arguments, interpretation_extension=None):
        original_valuation = self.interpretation.get(function, None)
        extended_valuation = interpretation_extension.get(function, None)
        if arguments is None:
            # Constant/Variable
            # Valuation 'extensions' override the originally provided valuation.
            valuation = original_valuation if extended_valuation is None else extended_valuation
            if valuation is None:
                raise VDPExceptions.PartialInterpretationException("The function you want is not interpreted by the model.")
            return valuation
        else:
            # Valuation 'extensions' override the originally provided valuation if there is a clash. Otherwise a union.
            valuation = {**original_valuation, **extended_valuation}
            if valuation is None:
                raise VDPExceptions.PartialInterpretationException("The function you want is not interpreted by the model.")
            value = valuation.get(arguments, None) if isinstance(valuation, dict) else valuation
            if value is None:
                raise VDPExceptions.PartialInterpretationException("Function does not interpret the given arguments.")
            return value

    # Returns the interpretation of a given term in the model.
    # A term is expected to be a list of lists, but this can be abstracted away into a class for terms later.
    # An interpretation 'extension' is an additional dictionary that can be given to interpret additional variables
    #  such as those bound by quantifiers.
    def interpret(self, term, interpretation_extension=None):
        if len(term) == 1:
            # Constant symbol
            return self.lookup_interpretation(term, None)
        else:
            eval_term_rec = [self.interpret(subterm, interpretation_extension) if isinstance(subterm, list) else subterm
                             for subterm in term[1:]]
            return self.lookup_interpretation(term[0], tuple(eval_term_rec), interpretation_extension)
