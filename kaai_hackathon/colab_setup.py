"""One-call setup for running these notebooks on Google Colab.

Every notebook starts by calling this. Outside Colab it does nothing at all, so the same
notebooks run unchanged on a cluster where the data is already on disk -- there is one set of
notebooks, not two, and no way for them to drift apart.

On Colab it does three things:

1. makes sure `kaai_hackathon` is importable,
2. mounts the public data bucket with gcsfuse, so `read_catalog` gets an ordinary filesystem
   path and nothing in the toolkit has to know about object storage,
3. sets CAMELS_HACKATHON_DATA and CAMELS_HACKATHON_PARAMS, which is all the notebooks read.

Usage, as the first cell of a notebook::

    !pip -q install git+https://github.com/xwzhang98/kaai-robust-inference-hackathon-2026
    from kaai_hackathon.colab_setup import setup; setup()
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

BUCKET = "kaai-hackathon-2026-data"
MOUNT = Path("/content/data")
REPO_URL = "https://github.com/xwzhang98/kaai-robust-inference-hackathon-2026"


def in_colab() -> bool:
    return "google.colab" in sys.modules


def _run(command: str, check: bool = True) -> int:
    print(f"$ {command}", flush=True)
    return subprocess.run(command, shell=True, check=check).returncode


def _install_gcsfuse() -> None:
    if shutil.which("gcsfuse"):
        return
    codename = subprocess.run("lsb_release -c -s", shell=True, capture_output=True,
                              text=True).stdout.strip()
    _run(f'echo "deb https://packages.cloud.google.com/apt gcsfuse-{codename} main" '
         "| sudo tee /etc/apt/sources.list.d/gcsfuse.list > /dev/null")
    _run("curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg "
         "| sudo apt-key add - 2>/dev/null")
    _run("sudo apt-get -qq update && sudo apt-get -qq install -y gcsfuse")


def mount_bucket(bucket: str = BUCKET, mount: Path = MOUNT) -> Path:
    """Mount the data bucket read-only and return the mount point.

    Read-only on purpose: nothing a notebook does should be able to write to the shared
    dataset. The file cache matters more than it looks -- h5py makes many small reads when it
    opens a file, and each one that misses the cache is a round trip to another region.
    """
    mount.mkdir(parents=True, exist_ok=True)
    if any(mount.iterdir()):
        print(f"{mount} is already mounted")
        return mount
    _install_gcsfuse()
    _run(f"gcsfuse --implicit-dirs -o ro --file-cache-max-size-mb=8192 {bucket} {mount}")
    return mount


def setup(bucket: str = BUCKET, mount: Path = MOUNT, authenticate: bool = True) -> None:
    """Prepare a Colab runtime. A no-op anywhere else."""
    if not in_colab():
        for name in ("CAMELS_HACKATHON_DATA", "CAMELS_HACKATHON_PARAMS"):
            if name not in os.environ:
                print(f"not on Colab and {name} is unset -- point it at your catalogs")
        return

    import kaai_hackathon
    print(f"kaai_hackathon {kaai_hackathon.__version__} from "
          f"{Path(kaai_hackathon.__file__).parent}")

    if authenticate:
        from google.colab import auth        # noqa: PLC0415  (Colab-only import)
        auth.authenticate_user()

    root = mount_bucket(bucket, mount)
    os.environ["CAMELS_HACKATHON_DATA"] = str(root / "catalogs")
    os.environ["CAMELS_HACKATHON_PARAMS"] = str(root / "params")

    suites = sorted(p.name for p in (root / "catalogs").iterdir()) \
        if (root / "catalogs").is_dir() else []
    print(f"\nCAMELS_HACKATHON_DATA   = {os.environ['CAMELS_HACKATHON_DATA']}")
    print(f"CAMELS_HACKATHON_PARAMS = {os.environ['CAMELS_HACKATHON_PARAMS']}")
    print(f"suites visible: {suites or 'NONE -- check that you have access to the bucket'}")

    try:
        import torch
        print(f"torch {torch.__version__}, GPU: "
              f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    except ImportError:
        pass
