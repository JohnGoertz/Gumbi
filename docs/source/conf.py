# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import pathlib as pl
sys.path.insert(0, os.path.abspath('../../gumbi/'))

PROJECT_ROOT = pl.Path(__file__).resolve().parent.parent.parent
VERSION = PROJECT_ROOT / "VERSION"

# -- Project information -----------------------------------------------------

project = 'Gumbi'
copyright = '2021, John Goertz'
author = 'John Goertz'

# The full version, including alpha/beta/rc tags
with open(VERSION, encoding="utf-8") as f:
    version = f.read().strip()
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    #'sphinxcontrib.rsvgconverter',
    #'numpydoc',
    'sphinx.ext.napoleon',
    #'sphinx_copybutton',
    'nbsphinx',
]
#numpydoc_show_class_members = True
numpydoc_class_members_toctree = False

autosummary_generate = True

# autodoc_default_options = {
#     'members': True,
#     'special-members': True,
#     'private-members': False,
#     'inherited-members': True,
#     'undoc-members': False,
#     'exclude-members': '__weakref__',
# }


nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_css_files = [
    "green_theme.css",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


