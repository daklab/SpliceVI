"""Build helper for the documentation site: fill in the parameter descriptions that SPLICEVI shares with scvi-tools.

Some SPLICEVI docstrings contain placeholders such as %(de_adata)s, which scvi-tools' decorators replace with the
shared descriptions when the package is imported. The site's docstring reader (griffe, used by mkdocstrings) reads the
source file, where only the placeholders are written; this extension gives it the filled-in docstring instead.
Registered in mkdocs.yml under the mkdocstrings options."""
import importlib
import inspect

import griffe


class FillInSharedDocstrings(griffe.Extension):
    """Use the runtime docstring of every function or method whose source docstring contains a %(...)s placeholder."""

    def on_function_instance(self, *, node, func, agent, **kwargs):
        if func.docstring is None or "%(" not in func.docstring.value:
            return

        # Walk from the module to the object: module.Class.method
        runtime_object = importlib.import_module(func.module.path)
        for name in func.path[len(func.module.path) + 1:].split("."):
            runtime_object = getattr(runtime_object, name)

        runtime_docstring = getattr(runtime_object, "__doc__", None)
        if runtime_docstring and "%(" not in runtime_docstring:
            func.docstring.value = inspect.cleandoc(runtime_docstring)
