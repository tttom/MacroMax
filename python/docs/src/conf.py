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
import pathlib
import sys
from datetime import datetime

import sphinx.ext.apidoc

import macromax

root_path = pathlib.Path(__file__).parent.parent.parent.absolute()
code_path = pathlib.Path(macromax.__file__).parent

sys.path.insert(0, root_path.as_posix())  # To generate documentation locally


# -- Project information -----------------------------------------------------

project = 'MacroMax'
author = 'Tom Vettenburg'
copyright = f'{datetime.now().year}, {author}'
version = macromax.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'm2r2',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # Used to write beautiful docstrings
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',  # Used to insert typehints into the final docs
    'sphinxcontrib.mermaid',  # Used to build graphs
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'sphinx.ext.imgconverter',
    'matplotlib.sphinxext.roles',  # to get mpltype text role for matplotlib
]

source_suffix = ['.rst', '.md']

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

autoclass_content = 'class'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'special-members': True,
    'inherited-members': True,
    'undoc-members': True,
    'exclude-members': '__abstractmethods__, __annotations__, __class_getitem__, __contains__, ' +
    '__dict__, __getitem__, __hash__, __module__, __orig_bases__, __parameters__, __repr__, ' +
    '__reversed__, __slots__, __static_attributes__, __str__, __subclasshook__, __weakref__',
    'show-inheritance': True,
}

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'coloredlogs': ('https://coloredlogs.readthedocs.io/en/latest/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'jax': ('https://docs.jax.dev/en/latest/', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest/', None),
}

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
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # ToC options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': -1,
    'includehidden': True,
    'titles_only': False
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autodoc_mock_imports = ['jax', 'torch', 'tensorflow']

# Building the API Documentation...
docs_path = root_path / 'docs'
apidoc_path = docs_path / 'src/api'  # a temporary directory
print(f'Building api-doc scaffolding in {apidoc_path}...')
sphinx.ext.apidoc.main(
    [
        '-f', '-d', '4', '-M',
        '-o', str(apidoc_path),
        str(code_path),
        f"{code_path}/utils/ft/ft*",  # apidocs fails on ft_implementation for some reason
    ]
)


# -- Options for EPUB3 output -------------------------------------------------
epub_cover = ('_static/total_internal_reflection.png', '')
