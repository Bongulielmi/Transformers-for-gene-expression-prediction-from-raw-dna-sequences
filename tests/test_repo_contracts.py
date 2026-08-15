"""Executable contracts for the rules stated in CLAUDE.md.

These tests turn CLAUDE.md from prose into enforced checks so the document
cannot silently drift away from the codebase.

They deliberately parse the modules with `ast` instead of importing them:
CI installs no ML dependencies (there is no requirements.txt yet, see
CLAUDE.md section 9), so `import tensorflow` would fail for reasons that
have nothing to do with the contract under test.
"""

import ast
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "Classes_py"
MODULES = sorted(MODULE_DIR.glob("*.py"))

# CLAUDE.md section 1: constructs that must never reach a .py module.
FORBIDDEN_MAGICS = (
    "%pylab",
    "%tensorflow_version",
    "%load_ext",
    "%matplotlib",
    "%%bash",
    "!pip",
    "!rm",
    "!wget",
)


def test_module_dir_is_populated():
    """Guard against the glob silently matching nothing."""
    assert MODULES, f"no modules found under {MODULE_DIR}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_module_parses_as_python(path):
    """CLAUDE.md section 0: Classes_py holds canonical *runnable* modules.

    A notebook magic makes a file unparseable, which the CI flake8 gate
    reports as E999. Parsing here pins that fix in place.
    """
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:  # pragma: no cover - failure path
        pytest.fail(f"{path.relative_to(REPO_ROOT)} is not valid Python: {exc}")


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_module_has_no_notebook_magics(path):
    """CLAUDE.md section 1 and section 7: no notebook-only constructs in modules."""
    offenders = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # a magic mentioned inside a comment is inert
        for magic in FORBIDDEN_MAGICS:
            if stripped.startswith(magic):
                offenders.append(f"line {lineno}: {stripped}")
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} contains notebook-only constructs:\n  "
        + "\n  ".join(offenders)
    )


def _class_defs(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def _method_names(class_node):
    return {n.name for n in class_node.body if isinstance(n, ast.FunctionDef)}


def test_datamanager_exposes_documented_split_accessors():
    """CLAUDE.md section 4: DataManager loads HDF5 into train/val/test splits."""
    classes = _class_defs(MODULE_DIR / "DataManager.py")
    assert "DataManager" in classes, "DataManager class not found"
    methods = _method_names(classes["DataManager"])
    for expected in ("get_train", "get_validation", "get_test"):
        assert expected in methods, f"DataManager.{expected}() is documented but missing"


def test_model_classes_build_their_model_in_init():
    """CLAUDE.md section 3: `_build_model()` is invoked in `__init__`."""
    expected_owners = {
        "Transformer.py": "projTransformer",
        "CNN1D.py": "projCNN1D",
        "TF_net.py": "projTFNet",
        "BioLSTM.py": "BioLSTM",
    }
    for filename, class_name in expected_owners.items():
        classes = _class_defs(MODULE_DIR / filename)
        assert class_name in classes, f"{class_name} not found in {filename}"
        methods = _method_names(classes[class_name])
        assert "_build_model" in methods, f"{class_name}._build_model() is missing"

        init = next(
            n for n in classes[class_name].body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        calls_build = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_build_model"
            for node in ast.walk(init)
        )
        assert calls_build, f"{class_name}.__init__ does not call self._build_model()"


def test_claude_md_only_references_paths_that_exist():
    """CLAUDE.md section 10: do not invent module names or directories.

    Only in-repo source paths are checked. `Dataset/` is deliberately
    excluded: section 8 requires that it never be committed.
    """
    claude_md = REPO_ROOT / "CLAUDE.md"
    assert claude_md.exists(), "CLAUDE.md is missing"

    text = claude_md.read_text(encoding="utf-8")
    candidates = {
        token
        for token in re.findall(r"`([^`\n]+)`", text)
        if token.startswith(("Classes_py/", "Classes/", ".github/"))
    }
    assert candidates, "no in-repo paths cited in CLAUDE.md - did the format change?"

    missing = []
    for token in sorted(candidates):
        matches = list(REPO_ROOT.glob(token)) if "*" in token else None
        if matches is not None:
            if not matches:
                missing.append(token)
        elif not (REPO_ROOT / token).exists():
            missing.append(token)

    assert not missing, (
        "CLAUDE.md references paths that do not exist:\n  " + "\n  ".join(missing)
    )
