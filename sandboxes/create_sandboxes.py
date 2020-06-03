# -*- coding: utf-8 -*-
"""
Create sandboxes for stormtrack, the feature tracking tool.

Sandboxes are test cases based on real data for testing beyond unit-like tests,
including runtime and memory profiling.

To ensure that dependencies (e.g., click) are available, it is recommended to
run this script using the virtual environment with testing requirements:

    $ cd <stormtrack>
    $ make install-test
    $ venv/bin/python sandboxes/create_cansboxes.py <...>

"""
# Standard library
import re
import ftplib
import logging as log
import shutil
from dataclasses import dataclass
from ftplib import FTP
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Union

# Third-party
import click


# Logging
log.basicConfig(format="%(message)s")


# Some defaults
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "template"
DEFAULT_DATA_PATH_FTP = "ftp://iacftp.ethz.ch/pub_read/ruestefa/stormtrack/sandboxes"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "dest", type=Path,
)
@click.option(
    "--skeleton-path",
    help="Path to sandbox seketon (scripts etc.), including 'sandboxes/'.",
    type=Path,
    default=DEFAULT_TEMPLATE_PATH,
)
@click.option(
    "--data-path-ftp",
    help="FPT path to data, including 'sandboxes/'.",
    type=str,
    default=DEFAULT_DATA_PATH_FTP,
)
@click.option(
    "--force/--no-force", "-f", help="Overwrite existing files.", default=False,
)
@click.option(
    "--link-run-scripts/--no-link-run-scripts",
    "-l",
    help="Create symlinks to template run scripts instead of copying them.",
    default=False,
)
@click.option(
    "--ftp-retry",
    help="Number of times a timed-out FTP download is retried.",
    type=int,
    default=5,
)
@click.option(
    "--dry/--no-dry", help="Don't copy any files.", default=False,
)
@click.option(
    "--dry-skeleton/--no-dry-skeleton",
    help=(
        "Show what would be done to create the sandbox skeleton (scripts etc.), but"
        " don't copy any files."
    ),
    default=False,
)
@click.option(
    "--dry-data/--no-dry-data",
    help=(
        "Show what data files would be copied, but don't copy any files."
        " (Still requires access to the FTP server where the files are stored.)"
    ),
    default=False,
)
@click.option(
    "-v", "vb", help="Pass once or more to increase the verbosity.", count=True,
)
def cli(
    dest,
    skeleton_path,
    data_path_ftp,
    force,
    link_run_scripts,
    ftp_retry,
    dry,
    dry_skeleton,
    dry_data,
    vb,
):
    """Create sandboxes at DEST."""
    if dest.name != "sandboxes":
        dest /= "sandboxes"
    if dry:
        dry_skeleton = True
        dry_data = True
    set_verbosity(vb)
    create_skeleton(
        skeleton_path, dest, link=link_run_scripts, force=force, dry=dry_skeleton, vb=vb
    )
    download_data(
        data_path_ftp, dest, force=force, n_retry=ftp_retry, dry=dry_data, vb=vb
    )


def set_verbosity(level):
    if level <= 0:
        log.getLogger().setLevel(log.WARNING)
    elif level == 1:
        log.getLogger().setLevel(log.INFO)
    else:
        log.getLogger().setLevel(log.DEBUG)


def create_skeleton(
    path: Path, dest: Path, *, force: bool, link: bool, dry: bool, vb: int
) -> None:
    """Create sandbox skeleton (scripts etc.)."""
    if dry:
        set_verbosity(max(1, vb))
    link_expr = r"\b.*\.sh\b" if link else None
    copy_tree(path, dest, force=force, link_expr=link_expr, dry=dry)
    set_verbosity(vb)


def download_data(
    path: str, dest: Path, *, force: bool, n_retry: int, dry: bool, vb: int
) -> None:
    """Download data files from FTP server."""
    if dry:
        set_verbosity(max(1, vb))
    ftp_path = FTPPath.from_str(path)
    ftp_path.download(
        dest, FTPPathDownloadConfig(force=force, n_retry=n_retry, dry=dry)
    )
    set_verbosity(vb)


def copy_tree(
    path: Path,
    dest: Path,
    force: bool = False,
    link_expr: Optional[Union[str, re.Pattern]] = None,
    dry: bool = False,
) -> None:
    """Copy a directory tree.

    Args:
        path: Root path of tree to copy.

        dest: Root path where the tree is copied.

        force (optional): Overwrite existing files.

        link_expr (optional): Files matching this regular expression in name are
            soft-linked rather than copied.

        dry (optional): Don't copy any files.

    """
    log.debug(f"copy tree: {path} -> {dest}")
    if isinstance(link_expr, str):
        link_expr = re.compile(link_expr)
    for node in path.glob("*"):
        node_dest = dest / node.name
        if node.is_dir():
            copy_tree(node, node_dest, force=force, link_expr=link_expr, dry=dry)
        else:
            copy_file(node, node_dest, force=force, link_expr=link_expr, dry=dry)


def copy_file(
    path: Path,
    dest: Path,
    force: bool = False,
    link_expr: Optional[re.Pattern] = None,
    dry: bool = False,
) -> None:
    """Copy (or link) a file to a destination."""
    if not dry:
        if dest.exists():
            if not force:
                raise FileExistsError(str(dest))
            dest.unlink()
        dest.parent.mkdir(exist_ok=True, parents=True)
    if link_expr is not None and link_expr.match(path.name):
        log.info(f"link file: {path} -> {dest}")
        if not dry:
            dest.symlink_to(path.absolute())
    else:
        log.info(f"copy file: {path} -> {dest}")
        if not dry:
            if dest.exists():
                dest.unlink()
            shutil.copyfile(path, dest)


class FTPDownloadError(Exception):
    """An error occurred during a downloade via FPT."""


@dataclass
class FTPPathDownloadConfig:
    n_retry: int = 5
    force: bool = False
    dry: bool = False


@dataclass
class FTPPath:
    host: str
    path: Path

    def __post_init__(self):
        download_config: FTPPathDownloadConfig = FTPPathDownloadConfig()
        self.ftp: Optional[FTP] = None

    @classmethod
    def from_str(cls, path: str, **kwargs) -> "FTPPath":
        """Create an FTPPath object from a string."""
        rx = r"\b(ftp://)?(?P<host>(\w+\.)+\w+)/(?P<path>.*)\b"
        match = re.match(rx, path)
        if not match:
            raise ValueError(f"path does not match '{rx}': {path}")
        path = Path("/" + match.group("path"))
        return cls(host=match.group("host"), path=path, **kwargs)

    def download(
        self, dest: Union[str, Path], config: Optional[FTPPathDownloadConfig] = None
    ) -> None:
        """Download files and folders under path from FTP host to dest."""
        if config is not None:
            self.download_config = config
        log.info(f"download {self.host}:{self.path} to {dest}")
        with FTP(self.host) as self.ftp:
            response = self.ftp.login("anonymous", "foo@bar.ch")
            log.info(f"connecting to FTP host {self.host}: {response}")
            self._download_dir(self.path, dest)
        self.ftp = None

    def _download_file(
        self, path: Path, dest: Path, force: Optional[bool] = None
    ) -> None:
        """Download a file."""
        if force is None:
            force = self.download_config.force
        log.info(f"download file: {self.host}:{path} -> {dest}")
        if not self.download_config.dry:
            if dest.exists() and not force:
                raise FileExistsError(str(dest))
            dest.parent.mkdir(exist_ok=True, parents=True)
            try:
                self.ftp.retrbinary("RETR " + path.name, open(dest, "wb").write)
            except Exception as error:
                raise FTPDownloadError(
                    {
                        "cls": type(error),
                        "msg": str(error),
                        "host": self.host,
                        "path": str(path),
                        "dest": str(dest),
                    },
                )

    def _download_dir(self, path: Path, dest: Path) -> None:
        """Download a directory recursively."""
        log.debug(f"download dir: {self.host}:{path} -> {dest}")
        cdir = self.ftp.pwd()
        self.ftp.cwd(str(path))
        for node in self.ftp.nlst():
            self._download_node(node, dest)
        self.ftp.cwd(cdir)

    def _download_node(self, node: str, dest: Path) -> None:
        """Download a file or directory."""
        path = Path(self.ftp.pwd()) / Path(node)
        log.debug(f"download node: {self.host}:{path} -> {dest}")
        dest = dest / Path(node)
        wd = self.ftp.pwd()
        try:
            self.ftp.cwd(str(path))
        except ftplib.error_perm:
            for i_try in range(self.download_config.n_retry):
                force = self.download_config.force if i_try == 0 else True
                try:
                    self._download_file(path, dest, force)
                except FTPDownloadError as error:
                    if issubclass(error.args[0]["cls"], TimeoutError):
                        if i_try == self.download_config.n_retry - 1:
                            raise FTPDownloadError(
                                "download failed after {i_try + 1) attemps"
                            ) from error
                        log.error(
                            f"download timed out; retrying ({i_try + 1}/"
                            f"{self.download_config.n_retry}) ..."
                        )
                        continue
                    raise
                break
        else:
            self._download_dir(path, dest)
            self.ftp.cwd(wd)


if __name__ == "__main__":
    cli()
