class VDPSymbol:
    """
    This class defines the concept of a generic 'symbol'. Symbols have a 'name' which is a string and a 'symbol_type'
    which denotes what part of the vocabulary they constitute, say object, relation, or label.
    Symbols will be used to represent elements in models as well as labels and relations.
    """

    def __init__(self, name, symbol_type=None):
        self.name = name
        self.symbol_type = symbol_type

    def symbol_is_instance(self, instance):
        return self.symbol_type == instance

    # Symbols with the same name must be considered equal.
    # To enforce symbol equality only when BOTH name and signature are equal, change __eq__ and __hash__ appropriately.
    # TODO (low): allow signature to determine equality as well.
    #  This change will generalise the model, but will have effects on the way that interpretations are keyed in
    #  VDPFOModel objects.
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, VDPSymbol):
            return self.name == other.name
        return NotImplemented

    # Custom symbol equality defined above must extend to usage as elements of mutable entities such as sets.
    # NOTE: IMPORTANT! In general implementation only makes sense if the symbol (or classes that inherit from it)
    #   DO NOT have mutable entities in their attributes.
    # TODO (low): opt for a different design to obtain persistent symbol identities that is more general.
    def __hash__(self):
        """Overrides the default implementation"""
        return hash(self.name)

    # Getter methods for attributes
    def get_name(self):
        return self.name

    def get_symbol_type(self):
        return self.symbol_type

    # No setter methods as we want Symbols to be immutable


class FOSort(VDPSymbol):
    """
    This class defines a 'sort' in the sense of multi-sorted FO. Each sort is simply a distinct name.
    """

    def __init__(self, name):
        super().__init__(name, 'FOSort')

# TODO (medium-low): Create classes that inherit from FOSort to create specialised sorts with interpreted functions.
#  One example is booleans.


class FOFunction(VDPSymbol):
    """
    This class defines a function. Each function has a name and a signature.
    The signature is a tuple of FOSort objects representing the type of the arguments.
    Singleton tuples denote constants and tuples with the final component as 'Bool' denote predicates.
    """

    # TODO (very low): replace strings in signature with
    #  or another symbol that denotes what kinds of arguments these are: object, label, etc.
    #  Important: these must be immutable entities or custom hashing for symbol ids will not work.
    def __init__(self, name, signature):
        self.signature = signature
        super().__init__(name, 'FOFunction')

    # Getter methods for attributes
    def get_relation_symbol_signature(self):
        return self.signature


# NOTE: classes representing the actual label-type or object-type elements
#  of a VDP FO model are present in vdpfomodel.py
