import unittest

# This file defines a first-order model as an FOModel object and performs various tests on it
# The model is as follows:
# Sorts: Object, Label
# Functions: On:(Object,Object,Bool), LabelOf:(Object,Label,Bool)
# Elements: Object-> {o1,o2,o3}, Label -> {cat, sofa}
# Interpretation: [LabelOf(o1,cat), LabelOf(o2,cat), LabelOf(o3,sofa), On(o1,o3)]. Tuples not specified are false


from vdp.vocabulary import FOSort, FOFunction
from vdp.fomodel import FOElement, FOModel
from vdp.exceptions import PartialInterpretationException


class FOModelTest(unittest.TestCase):

    def setUp(self):
        # Create the vocabulary
        # Sorts
        vdpobject = FOSort('Object')
        vdplabel = FOSort('Label')
        vdpbool = FOSort('Bool')

        # Constants
        cat = FOFunction('cat', tuple([vdplabel]))
        sofa = FOFunction('sofa', tuple([vdplabel]))
        # Relations
        on = FOFunction('On', (vdpobject, vdpobject, vdpbool))
        labelof = FOFunction('LabelOf', (vdpobject, vdplabel, vdpbool))
        # No functions in this signature.
        # Make the vocabulary as a pair: ([list of sorts],[list of constants/functions/relations]).
        vocabulary = ([vdpobject, vdplabel], {cat, sofa, on, labelof})

        # Create the elements of each sort
        o1 = FOElement('o1', vdpobject)
        o2 = FOElement('o2', vdpobject)
        o3 = FOElement('o3', vdpobject)

        cat_label = FOElement('cat_label', vdplabel)
        sofa_label = FOElement('sofa_label', vdplabel)

        elements = {'Object': {o1, o2, o3}, 'Label': {cat_label, sofa_label}}

        # Define the interpretation
        labelconst_interpretation = {cat: cat_label, sofa: sofa_label}
        labelof_interpretation = {labelof: {(o1, cat_label): True, (o2, cat_label): True, (o3, sofa_label): True,
                                            (o1, sofa_label): False, (o2, sofa_label): False, (o3, cat_label): False}}
        on_interpretation = {on: {(o1, o3): True, (o2, o3): True,
                                  (o1, o2): False, (o3, o1): False, (o3, o2): False}}
        interpretation = {**labelconst_interpretation, **labelof_interpretation, **on_interpretation}

        # Create the FO Model
        model = FOModel(vocabulary, elements, interpretation)

        # Set variables to use in tests
        self.vdpobject, self.vdplabel = vdpobject, vdplabel
        self.cat, self.sofa = cat, sofa
        self.on, self.labelof = on, labelof
        self.o1, self.o2, self.o3 = o1, o2, o3
        self.model = model

    # Test for the value of various terms

    def test_normal_term(self):
        # Variables
        x = FOElement('x', self.vdpobject)
        y = FOElement('y', self.vdpobject)
        normal_term = self.model.interpret(term=[self.on, [x], [y]], interpretation_extension={x: self.o1, y: self.o2})
        self.assertFalse(normal_term)

    def test_term_with_constant(self):
        # Variables
        x = FOElement('x', self.vdpobject)
        term_with_constant = self.model.interpret(term=[self.labelof, [x], [self.cat]], interpretation_extension={x: self.o1})
        self.assertTrue(term_with_constant)

    def test_invalid_interpretation(self):
        # The error in this case is that o1 is not a constant in the logic and must always be 'interpreted' by
        # a variable, as can be seen from the other tests.
        # Variables
        x = FOElement('x', self.vdpobject)
        with self.assertRaises(PartialInterpretationException):
            self.model.interpret(term=[self.on, [self.o1], [x]], interpretation_extension={x: self.o2})

    def test_invalid_term(self):
        # Malformed term: well-formed terms must always be a list of lists
        # Variables
        x = FOElement('x', self.vdpobject)
        y = FOElement('y', self.vdpobject)
        with self.assertRaises(Exception):
            self.model.interpret(term=[self.on, x, y], interpretation_extension={x: self.o2})


if __name__ == '__main__':
    unittest.main()
