import unittest
import math
from interface.agents import ReasoningAgent

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.agent = ReasoningAgent(None)

    def test_calculate_simple_math(self):
        self.assertEqual(self.agent._tool_calculate('1 + 1'), '2')
        self.assertEqual(self.agent._tool_calculate('5 * 3'), '15')
        self.assertEqual(self.agent._tool_calculate('10 / 2'), '5.0')
        self.assertEqual(self.agent._tool_calculate('10 - 2'), '8')
        self.assertEqual(self.agent._tool_calculate('2 ** 3'), '8')
        self.assertEqual(self.agent._tool_calculate('10 % 3'), '1')
        self.assertEqual(self.agent._tool_calculate('10 // 3'), '3')
        self.assertEqual(self.agent._tool_calculate('-5'), '-5')
        self.assertEqual(self.agent._tool_calculate('+5'), '5')

    def test_calculate_functions(self):
        self.assertEqual(self.agent._tool_calculate('abs(-10)'), '10')
        self.assertEqual(self.agent._tool_calculate('round(3.14159, 2)'), '3.14')
        self.assertEqual(self.agent._tool_calculate('min(1, 2, 3)'), '1')
        self.assertEqual(self.agent._tool_calculate('max(1, 2, 3)'), '3')
        self.assertEqual(self.agent._tool_calculate('sum([1, 2, 3])'), '6')
        self.assertEqual(self.agent._tool_calculate('pow(2, 3)'), '8.0')

    def test_calculate_math_module(self):
        self.assertEqual(self.agent._tool_calculate('sqrt(16)'), '4.0')
        self.assertEqual(self.agent._tool_calculate('cos(0)'), '1.0')
        self.assertEqual(self.agent._tool_calculate('log10(100)'), '2.0')

        # Test constants
        self.assertTrue(self.agent._tool_calculate('pi').startswith('3.14'))
        self.assertTrue(self.agent._tool_calculate('e').startswith('2.71'))

    def test_calculate_malicious_inputs(self):
        # Prevent access to __builtins__ and system commands
        res = self.agent._tool_calculate('__import__("os").system("echo 1")')
        self.assertTrue('Erreur' in res)

        # Prevent accessing attributes
        res = self.agent._tool_calculate('[].__class__')
        self.assertTrue('Erreur' in res)

        # Prevent getattr, setattr, etc
        res = self.agent._tool_calculate('getattr(__builtins__, "eval")')
        self.assertTrue('Erreur' in res)

        # Prevent function declaration or any complex statements
        res = self.agent._tool_calculate('def foo(): pass')
        self.assertTrue('Erreur' in res)

        # Unapproved functions
        res = self.agent._tool_calculate('print("hello")')
        self.assertTrue('Erreur' in res)

if __name__ == '__main__':
    unittest.main()
