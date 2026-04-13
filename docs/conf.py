import os
import sys

project = "Coalsack"
copyright = "2026, Coalsack Contributors"
author = "Coalsack Contributors"

extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]

breathe_projects = {"coalsack": "_doxygen/xml"}
breathe_default_project = "coalsack"

html_theme = "sphinx_rtd_theme"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress Breathe duplicate_declaration warnings caused by common method names
# (e.g. serialize, get_proc_name) appearing in multiple unrelated classes within a group.
suppress_warnings = ["duplicate_declaration.cpp"]
