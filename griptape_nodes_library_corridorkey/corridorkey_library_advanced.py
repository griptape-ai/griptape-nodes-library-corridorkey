import logging
import subprocess
import sys
from pathlib import Path

import pygit2
from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logger = logging.getLogger("corridorkey_library")


class CorridorKeyLibraryAdvanced(AdvancedNodeLibrary):
    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Loading '{library_data.name}' library...")
        submodule_path = self._init_submodule()
        if not self._is_installed(submodule_path):
            self._install_from_requirements(submodule_path)
            self._install_pip_package(submodule_path)
            self._write_installed_sentinel(submodule_path)
        # Always re-apply sys.path so BiRefNetModule is importable. sys.path
        # mutations do not persist across engine restarts, and adding the same
        # path twice is a harmless no-op via the membership check below.
        self._install_syspath(submodule_path)
        # The engine loads each node .py file as a standalone dynamic module by
        # file path, not as part of an installed `griptape_nodes_library_corridorkey`
        # package, so `from griptape_nodes_library_corridorkey import corridorkey_common`
        # would fail to resolve the package. Add the repo root (this library's parent
        # directory, which contains the `griptape_nodes_library_corridorkey/` package
        # with its own __init__.py) to sys.path so that import works as written.
        self._install_own_syspath()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Finished loading '{library_data.name}' library")

    def _get_library_root(self) -> Path:
        return Path(__file__).parent

    def _get_venv_python_path(self) -> Path:
        root = self._get_library_root()
        if sys.platform == "win32":
            return root / ".venv" / "Scripts" / "python.exe"
        return root / ".venv" / "bin" / "python"

    def _update_submodules_recursive(self, repo_path: Path) -> None:
        repo = pygit2.Repository(str(repo_path))
        repo.submodules.update(init=True)
        for sub in repo.submodules:
            sub_path = repo_path / sub.path
            if sub_path.exists() and (sub_path / ".git").exists():
                self._update_submodules_recursive(sub_path)

    def _init_submodule(self) -> Path:
        library_root = self._get_library_root()
        submodule_dir = library_root / "CorridorKey"
        if submodule_dir.exists() and any(submodule_dir.iterdir()):
            logger.info("Submodule already initialized")
            return submodule_dir
        self._update_submodules_recursive(library_root.parent)
        if not submodule_dir.exists() or not any(submodule_dir.iterdir()):
            raise RuntimeError(f"Submodule init failed: {submodule_dir}")
        logger.info("Submodule initialized successfully")
        return submodule_dir

    def _ensure_pip(self) -> None:
        venv_python = self._get_venv_python_path()
        result = subprocess.run([str(venv_python), "-m", "pip", "--version"], capture_output=True)
        if result.returncode == 0:
            return
        subprocess.check_call([str(venv_python), "-m", "ensurepip", "--upgrade"])

    def _get_submodule_commit(self, submodule_path: Path) -> str:
        """Return the HEAD commit SHA of the submodule (the version pinned by the library author)."""
        repo = pygit2.Repository(str(submodule_path))
        return str(repo.head.target)

    def _get_installed_sentinel(self) -> Path:
        return self._get_library_root() / ".installed_commit"

    def _write_installed_sentinel(self, submodule_path: Path) -> None:
        self._get_installed_sentinel().write_text(self._get_submodule_commit(submodule_path))

    def _is_installed(self, submodule_path: Path) -> bool:
        """Return True only if CorridorKeyModule is importable AND was installed from the currently-pinned commit.

        This ensures that when a new library version ships with a different submodule commit,
        the package is reinstalled rather than reusing a stale installation. The sys.path-only
        BiRefNetModule is intentionally not checked here because sys.path is re-applied on every
        load by `_install_syspath` regardless of this method's return value.
        """
        venv_python = self._get_venv_python_path()
        result = subprocess.run(
            [str(venv_python), "-c", "import CorridorKeyModule"],
            capture_output=True,
        )
        if result.returncode != 0:
            return False
        sentinel = self._get_installed_sentinel()
        if not sentinel.exists():
            return False
        return sentinel.read_text().strip() == self._get_submodule_commit(submodule_path)

    def _install_from_requirements(self, submodule_path: Path) -> None:
        """Install dependencies from the submodule's requirements.txt.

        This preserves platform markers, version pins, and extra-index-url
        directives exactly as the model author intended.

        Uses --no-build-isolation so that packages requiring torch at build time
        (e.g., auto_gptq, flash-attn) can find the torch already installed in the venv.
        Without this flag, pip creates an isolated build environment that doesn't
        see the venv's packages, causing "No module named 'torch'" build failures.
        """
        requirements_file = submodule_path / "requirements.txt"
        if not requirements_file.exists():
            logger.info("No requirements.txt found in submodule, skipping")
            return
        venv_python = self._get_venv_python_path()
        self._ensure_pip()
        logger.info(f"Installing requirements from {requirements_file}...")
        subprocess.check_call(
            [str(venv_python), "-m", "pip", "install", "--no-build-isolation", "-r", str(requirements_file)]
        )
        logger.info("Requirements installed successfully")

    def _install_pip_package(self, submodule_path: Path) -> None:
        """Install the submodule as a Python package (--no-deps since pip_dependencies handled deps).

        This installs the packages listed in the submodule's hatch wheel (CorridorKeyModule,
        gvm_core, VideoMaMaInferenceModule). It does NOT install BiRefNetModule, which lives in
        the submodule root but is not in the wheel's package list -- BiRefNetModule is exposed
        via `_install_syspath` instead.
        """
        venv_python = self._get_venv_python_path()
        self._ensure_pip()
        logger.info(f"Installing package from {submodule_path}...")
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "--no-deps", str(submodule_path)])
        logger.info("Package installed successfully")

    def _install_syspath(self, submodule_path: Path) -> None:
        """Add the submodule root to sys.path so BiRefNetModule is importable.

        BiRefNetModule lives at <submodule>/BiRefNetModule but is not in the hatch wheel's
        `packages` list, so `pip install --no-deps` does NOT copy it into site-packages.
        Path-injecting the submodule root makes `import BiRefNetModule` resolve to the
        in-tree directory.
        """
        if str(submodule_path) not in sys.path:
            sys.path.insert(0, str(submodule_path))
            logger.info(f"Added {submodule_path} to sys.path for BiRefNetModule")

    def _install_own_syspath(self) -> None:
        """Add the repo root to sys.path so `griptape_nodes_library_corridorkey` is importable.

        Node files are loaded by the engine as standalone dynamic modules by file
        path, not via a real installed `griptape_nodes_library_corridorkey` package,
        so `from griptape_nodes_library_corridorkey import corridorkey_common` would
        otherwise fail to resolve the package. Path-injecting the repo root (which
        contains the `griptape_nodes_library_corridorkey/` package, with its own
        __init__.py) makes that import resolve as written, keeping the shared helper
        module properly namespaced instead of a bare top-level name.
        """
        repo_root = str(self._get_library_root().parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
            logger.info(f"Added {repo_root} to sys.path for griptape_nodes_library_corridorkey")
