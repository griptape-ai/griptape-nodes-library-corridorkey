import logging
import subprocess
import sys
from pathlib import Path

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logger = logging.getLogger("corridorkey_library")


class CorridorKeyLibraryAdvanced(AdvancedNodeLibrary):
    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Loading '{library_data.name}' library...")
        # The CorridorKey package itself is declared in pip_dependencies_exec and installed by
        # the engine. The submodule survives for BiRefNetModule alone, which upstream excludes
        # from its wheel's `packages` list, so no install can deliver it -- only a source tree
        # on sys.path can.
        submodule_path = self._init_submodule()
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

    def _init_submodule(self) -> Path:
        library_root = self._get_library_root()
        submodule_dir = library_root / "CorridorKey"
        if submodule_dir.exists() and any(submodule_dir.iterdir()):
            logger.info("Submodule already initialized")
            return submodule_dir
        # The git CLI rather than pygit2: the engine dropped pygit2 (its bundled TLS trust
        # store breaks on some platforms) and requires git on PATH, so it is the one tool
        # guaranteed to be here.
        subprocess.check_call(["git", "-C", str(library_root.parent), "submodule", "update", "--init", "--recursive"])
        if not submodule_dir.exists() or not any(submodule_dir.iterdir()):
            raise RuntimeError(f"Submodule init failed: {submodule_dir}")
        logger.info("Submodule initialized successfully")
        return submodule_dir

    def _install_syspath(self, submodule_path: Path) -> None:
        """Add the submodule root to sys.path so BiRefNetModule is importable.

        BiRefNetModule lives at <submodule>/BiRefNetModule but is not in the upstream wheel's
        `packages` list, so installing the CorridorKey package does NOT copy it into
        site-packages. Path-injecting the submodule root makes `import BiRefNetModule`
        resolve to the in-tree directory.
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
