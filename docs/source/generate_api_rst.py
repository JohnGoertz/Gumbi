import inspect
import os
import pathlib as pl
from datetime import datetime

import gumbi

# TODO: Refactor generate_module_rst.py and document

RST_PATH = "source/"
RST_LEVEL_SYMBOLS = ["=", "-", "~", '"', "'", "^"]

file_header = """{headerline}
{name}
{headerline}

.. THIS IS AN AUTOGENERATED RST FILE
.. GENERATED BY `generate_rst.py`
.. DATE: {date}

"""

automodule = """
{name}
{headerline}
.. automodule:: {name}

"""

autofunction = """
{name}
{headerline}
.. autofunction:: {name}

"""

class_autosummary = """
.. rubric:: Classes
.. autosummary::
   :toctree: {toctree}

"""

functions_autosummary = """
.. currentmodule:: {name}

.. rubric:: Functions

.. autosummary::

"""

IGNORE_MODULES = {"gumbi.utils.misc"}

DATE_STRING = datetime.strftime(datetime.now(), "%d/%m/%y")


def set_global_path(path):
    global RST_PATH
    RST_PATH = pl.Path(path)


class Node:
    """Builds up the branches of the package"""

    def __init__(self, node, level):
        self.level = level
        self._node = node
        self.name = node.__name__
        self.nickname = self.name.split(".")[-1]
        self.id = id(node)
        self.ismodule = inspect.ismodule(self._node)
        self.isclass = inspect.isclass(self._node)
        self.isfunction = inspect.isfunction(self._node)
        self.public_attributes = [
            getattr(node, a) for a in dir(node) if not a.startswith("_")
        ]
        if self.ismodule:
            self.parent = ".".join(node.__name__.split(".")[:-1])
            self.fullname = self.name
        else:
            self.parent = node.__module__
            self.fullname = f"{self.parent}.{self.name}"

    def __repr__(self):
        return f"Node({self._node!r})"

    def is_ancestor_of(self, node):
        return self.name in node.parent

    def is_descendent_of(self, node):
        return node.name in self.parent

    @staticmethod
    def ignore(node):
        try:
            if inspect.ismodule(node):
                name = node.__name__
                parent = ".".join(name.split(".")[:-1])
            else:
                name = ""
                parent = node.__module__
        except AttributeError:
            return True

        relevant = "gumbi" in parent
        marked = any(module in parent or module in name for module in IGNORE_MODULES)
        return marked or not relevant

    def child(self, typ):
        if not self.ismodule:
            return []
        child_list = []
        for child in self.public_attributes:
            if not self.ignore(child):
                new_node = Node(child, self.level + 1)
                is_typ = {
                    "modules": new_node.ismodule,
                    "classes": new_node.isclass,
                    "functions": new_node.isfunction,
                }[typ]
                if is_typ and new_node.is_descendent_of(self):
                    child_list.append(new_node)
        return child_list

    @property
    def children(self):
        return self.child("modules") + self.child("classes") + self.child("functions")

    @property
    def child_ids(self):
        return {
            child.id
            for child in self.child("modules")
            + self.child("classes")
            + self.child("functions")
        }

    def modules_not_in(self, old: set):
        return [child for child in self.child("modules") if child.id not in old]

    def classes_not_in(self, old: set):
        return [child for child in self.child("classes") if child.id not in old]

    def functions_not_in(self, old: set):
        return [child for child in self.child("functions") if child.id not in old]

    def interesting(self, old: set):
        return any(self.child_ids - old)

    def interesting_modules(self, old: set):
        return [
            module for module in self.modules_not_in(old) if module.interesting(old)
        ]


def write_rst(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def traverse(node, bag):
    """Recursively step through this node's children to write each rst file"""

    if node.id in bag:
        # Been there, done that
        return

    header = file_header.format(
        name=node.name,
        headerline=RST_LEVEL_SYMBOLS[node.level] * len(node.name),
        date=DATE_STRING,
    )

    file_content = ""

    if any(node.interesting_modules(bag)):
        children_are_modules = [
            child.ismodule for child in node.children if child.interesting(bag)
        ]
        grandchildren_are_functions = [
            child.isfunction
            for module in node.child("modules")
            for child in module.children
        ]

        if all(grandchildren_are_functions) and all(children_are_modules):
            # This page will just be a toctree pointing to the submodules.
            file_content += ".. rubric:: Submodules\n.. toctree::\n\n"

    for module in node.interesting_modules(bag):

        children_are_functions = [child.isfunction for child in module.children]
        other_new_children = [
            child
            for child in module.classes_not_in(bag) + module.interesting_modules(bag)
        ]

        if all(children_are_functions):
            # This module will get its own index page with function descriptions. This index page will have an
            # autosummary at the top (header) followed by autofunction descriptions (content).
            index_header = file_header.format(
                name=module.name,
                headerline=RST_LEVEL_SYMBOLS[node.level] * len(module.name),
                date=DATE_STRING,
            )
            index_header += functions_autosummary.format(name=module.name)
            index_content = ""

            for grandchild in module.functions_not_in(bag):
                # Add this function to the autosummary
                index_header += f"   {grandchild.name}\n"
                # Add this function's autofunction description
                headerline = RST_LEVEL_SYMBOLS[module.level] * len(grandchild.fullname)
                index_content += autofunction.format(
                    name=grandchild.fullname, headerline=headerline
                )
                bag.add(grandchild.id)

            # Write the new module index page
            path = RST_PATH / pl.Path(*module.name.split(".")) / "index.rst"
            write_rst(path, index_header + "\n\n" + index_content)
            bag.add(module.id)

            # Point to the new module index page
            file_content += f"   {module.nickname}/index\n"

        elif any(other_new_children):
            # There's interesting stuff to be had, add an automodule description of this module to the file
            file_content += automodule.format(
                name=module.name,
                headerline=RST_LEVEL_SYMBOLS[node.level] * len(module.name),
            )

        else:
            # Got nothing interesting for us? Move along...
            continue

        new_classes = module.classes_not_in(bag)
        if any(new_classes):
            # Create a section for classes, and tell autosummary to generate new files for these classes
            file_content += class_autosummary.format(toctree=module.nickname)

            for grandchild in new_classes:
                file_content += f"   {grandchild.name}\n"
                bag.add(grandchild.id)

        if any(module.interesting_modules(bag)):
            # If there's anything further to document, place the reference to the yet-to-be-written index file in a
            # "Submodules" section
            module_header = (
                "\n.. rubric:: Submodules\n.. toctree::\n   :maxdepth: 2\n\n"
            )
            module_content = f"   {module.nickname}/index\n"
            file_content += module_header + module_content

        bag.add(module)

    if file_content != "":
        # If we've found anything worthwhile, write this file
        path = RST_PATH / pl.Path(*node.name.split(".")) / "index.rst"
        write_rst(path, header + file_content)
        bag.add(node)

    for module in node.child("modules"):
        # Recursion is fun is Recursion
        traverse(module, bag)


if __name__ == "__main__":
    set_global_path(os.path.dirname(os.path.realpath(__file__)))
    Gumbi = Node(gumbi, 0)
    empty_bag = set()
    traverse(Gumbi, empty_bag)
