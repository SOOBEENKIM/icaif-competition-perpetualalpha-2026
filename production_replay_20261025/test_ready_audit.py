import ast
import unittest
from .run_ready_audit import strip_absolute_waits


class SchedulingOnlyTests(unittest.TestCase):
    def test_keeps_dependencies_and_evaluation_statements(self):
        source='''def run():
    check()
    while time.time()<END+1200:
        save(status,{"state":"waiting"})
        time.sleep(30)
    while not ready():
        time.sleep(10)
    evaluate(start,END)
'''
        tree,removed=strip_absolute_waits(source)
        self.assertEqual(len(removed),1)
        transformed=ast.unparse(tree)
        self.assertIn('while not ready():',transformed)
        self.assertIn('evaluate(start, END)',transformed)
        self.assertIn('check()',transformed)

    def test_refuses_to_strip_evaluation_logic(self):
        with self.assertRaisesRegex(ValueError,'evaluation work'):
            strip_absolute_waits('def run():\n    while time.time()<END+10:\n        trade()\n')


if __name__=='__main__':unittest.main()
