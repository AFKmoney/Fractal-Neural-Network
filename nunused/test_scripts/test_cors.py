import unittest
from pathlib import Path
import sys

# Mocking modules that might be missing to allow importing app for static analysis
class Mock:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return Mock()
    def __call__(self, *args, **kwargs): return Mock()

# We only need to check if the middleware is present in the FastAPI app
# If we can't import the app because of missing deps, we can use ast to parse the file

import ast

class CORSChecker(ast.NodeVisitor):
    def __init__(self):
        self.has_cors_import = False
        self.has_cors_middleware = False
        self.allowed_origins = []

    def visit_ImportFrom(self, node):
        if node.module == 'fastapi.middleware.cors':
            for alias in node.names:
                if alias.name == 'CORSMiddleware':
                    self.has_cors_import = True
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'add_middleware':
            if any(isinstance(arg, ast.Name) and arg.id == 'CORSMiddleware' for arg in node.args):
                self.has_cors_middleware = True
                for kw in node.keywords:
                    if kw.arg == 'allow_origins' and isinstance(kw.value, ast.List):
                        self.allowed_origins = [ast.literal_eval(elt) for elt in kw.value.elts]
        self.generic_visit(node)

class TestCORSConfiguration(unittest.TestCase):
    def test_cors_middleware_is_added(self):
        app_path = Path(__file__).parent / "app.py"
        with open(app_path, "r") as f:
            tree = ast.parse(f.read())

        checker = CORSChecker()
        checker.visit(tree)

        self.assertTrue(checker.has_cors_import, "CORSMiddleware import missing")
        self.assertTrue(checker.has_cors_middleware, "CORSMiddleware not added to app")

        expected_origins = [
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
        for origin in expected_origins:
            self.assertIn(origin, checker.allowed_origins)

if __name__ == "__main__":
    unittest.main()
