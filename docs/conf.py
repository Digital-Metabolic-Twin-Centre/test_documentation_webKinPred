from datetime import datetime

project = "Open Kinetics Predictor"
author = "Digital Metabolic Twin Centre"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autoapi.extension",
]

exclude_patterns = [
    "_build",
    "build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom-wide.css"]
html_logo = "_static/img/logo.png"
html_favicon = "_static/img/favicon.ico"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "logo_only": False,
}
html_show_sphinx = False

suppress_warnings = [
    "autoapi",
]

autoapi_dirs = [
    "../api",
    "../webKinPred",
    "../db_models",
    "../tools/seqmap",
]
autoapi_add_toctree_entry = True
autoapi_options = [
]
autoapi_ignore = [
    "*/__pycache__/*",
    "*/migrations/*",
    "*/tests/*",
    "../api/tests/*",
    "../api/models.py",
    "../api/urls.py",
    "../api/urls_v1.py",
    "../api/views/v1_views.py",
    "../api/prediction_engines/generic_subprocess.py",
    "../api/utils/api_auth.py",
    "../api/utils/convert_to_mol.py",
    "../api/utils/validation_utils.py",
    "../tests/*",
    "../tools/memory_tests/*",
    "../tools/gpu_embed_service/*",
    "../tools/seqmap/test_*",
    "../tools/seqmap/utils/migrate_json_to_sqlite.py",
]


def autoapi_skip_member(app, what, name, obj, skip, options):
    return skip


def setup(app):
    app.connect("autoapi-skip-member", autoapi_skip_member)

    try:
        from astroid.exceptions import TooManyLevelsError
        from autoapi import _astroid_utils

        original_get_full_import_name = _astroid_utils.get_full_import_name

        def safe_get_full_import_name(module_node, level):
            try:
                return original_get_full_import_name(module_node, level)
            except TooManyLevelsError:
                partial_name = None
                if isinstance(level, str):
                    partial_name = level
                elif hasattr(module_node, "names"):
                    for import_name, imported_as in module_node.names:
                        partial_name = imported_as or import_name
                        if partial_name:
                            break

                module_name = getattr(module_node, "modname", "") or ""
                if module_name and partial_name:
                    return f"{module_name}.{partial_name}"
                return partial_name or module_name or "__autoapi_unresolved__"

        _astroid_utils.get_full_import_name = safe_get_full_import_name
    except (ImportError, AttributeError):
        pass
