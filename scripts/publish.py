# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "build",
#   "twine",
# ]
# ///
import glob
import itertools
import subprocess
from pathlib import Path


def publish():
    subprocess.run(["pyproject-build", "."], check=True)
    dists = list(
        itertools.chain.from_iterable(
            Path("dist").glob(dist_glob) for dist_glob in ("*.tar.gz", "*.whl")
        )
    )
    subprocess.run(["twine", "upload"] + dists, check=True)


if __name__ == "__main__":
    publish()
