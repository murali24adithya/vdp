class Symbol:
    """
    This class defines the concept of a generic 'symbol'. Symbols have a 'name' which is a string and a 'symbol_type'
    which denotes what part of the vocabulary they constitute, say object, relation, or label.
    Symbols will be used to represent elements in models as well as labels and relations.
    """

    def __init__(self, name, symbol_type=None):
        self.name = name
        self.symbol_type = symbol_type

    def symbol_is_instance(self, instance):
        """
        Function to check whether the current Symbol object is an instance of another Symbol kind
        higher in the hierarchy.
        It takes either a string or another Symbol object to check against the current object. If the argument is of
        neither type, it returns None.
        NOTE: The Symbol object to be compared against must be the object that represents the symbol_type class of
        objects, not another object whose symbol_type is the same as the current object.
        :param instance: string or Symbol
        :return: bool or None
        """
        if isinstance(instance, str):
            return self.symbol_type == instance
        elif isinstance(instance, Symbol):
            return self.symbol_type == instance.get_name()

    # Symbols with the same name must be considered equal.
    # To enforce symbol equality only when BOTH name and signature are equal, change __eq__ and __hash__ appropriately.
    # TODO (low): allow signature to determine equality as well.
    #  This change will generalise the model, but will have effects on the way that interpretations are keyed in
    #  VDPFOModel objects.
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Symbol):
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


class FOSort(Symbol):
    """
    This class defines a 'sort' in the sense of multi-sorted FO. Each sort is simply a distinct name.
    """

    def __init__(self, name):
        super().__init__(name, 'FOSort')


# TODO (medium-low): Create classes that inherit from FOSort to create specialised sorts with interpreted functions.
#  One example is booleans.


class FOFunction(Symbol):
    """
    This class defines a function. Each function has a name and a signature.
    The signature is a tuple of FOSort objects representing the type of the arguments.
    Singleton tuples denote constants and tuples with the final component as 'Bool' denote predicates.
    """

    def __init__(self, name, signature):
        self.signature = signature
        super().__init__(name, 'FOFunction')

    # Getter methods for attributes
    def get_relation_symbol_signature(self):
        return self.signature


# NOTE: classes representing the actual label-type or object-type elements
#  of an FO model are present in fomodel.py
