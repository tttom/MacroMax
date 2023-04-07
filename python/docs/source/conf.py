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
from datetime import datetime
from pathlib import Path
import sphinx.ext.apidoc
import sys

sys.path.insert(0, Path(__file__).parent.parent.absolute().as_posix())  # To generate documentation locally
# ReadTheDocs seems to require a combination of the following.
sys.path.insert(0, Path(__file__).parent.parent.parent.absolute().as_posix())
sys.path.insert(0, Path(__file__).parent.parent.absolute().as_posix())
sys.path.insert(0, Path.cwd().absolute().as_posix())
sys.path.insert(0, Path.cwd().parent.absolute().as_posix())
sys.path.insert(0, Path.cwd().parent.parent.absolute().as_posix())
sys.path.insert(0, Path.cwd().parent.parent.parent.absolute().as_posix())


# -- Project information -----------------------------------------------------

project = 'MacroMax'
author = 'Tom Vettenburg'
copyright = f'{datetime.now().year}, {author}'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['m2r2',  # or m2r
              'sphinx.ext.duration',
              'sphinx.ext.doctest',
              'sphinx.ext.autodoc',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',  # Used to write beautiful docstrings
              'sphinx_autodoc_typehints',  # Used to insert typehints into the final docs
              'sphinxcontrib.mermaid',  # Used to build graphs
              'sphinx.ext.intersphinx',
              'sphinx_rtd_theme',
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
    'exclude-members': '__dict__, __weakref__, __abstractmethods__, __annotations__, __parameters__, __module__, __getitem__, __str__, __repr__, __hash__, ' +
                       '__slots__, __orig_bases__, __subclasshook__, __class_getitem__, __contains__, __reversed__',  # , __eq__, __add__, __sub__, __neg__, __mul__, __imul__, __matmul__, __div__, __idiv__, __rdiv__, __truediv__',
    'show-inheritance': True,
}

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/', None),
                       'matplotlib': ('https://matplotlib.org/stable/', None),
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
    'sidebar_collapse': False,
    'show_powered_by': False,
}
html_theme_options = {
    'logo_only': False,
    'display_version': True,
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

source_suffix = ['.rst', '.md']

autodoc_mock_imports = ['torch', 'tensorflow']

# Building the API Documentation...
code_path = Path(__file__).parent.parent.parent.resolve()
docs_path = code_path / 'docs'
apidoc_path = docs_path / 'source/api'  # a temporary directory
html_output_path = docs_path / 'build/html'
print(f'Building api-doc scaffolding in {apidoc_path}...')
sphinx.ext.apidoc.main(['-f', '-d', '4', '-M',
                        '-o', f'{apidoc_path}',
                        f"{code_path / 'macromax'}"]
                       )
