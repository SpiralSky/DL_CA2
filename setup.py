import os
import shutil
import subprocess
import sys


UV_ENV = {
    "UV_PYTHON_INSTALL_DIR": "/workspace/.uv-python",
    "UV_CACHE_DIR": "/workspace/.uv-cache",
    "UV_LINK_MODE": "copy",
}


def configure_uv() -> None:
    """Configure uv paths for persistent RunPod volumes."""
    for key, value in UV_ENV.items():
        os.environ.setdefault(key, value)

    print("Configured uv:")
    for key in UV_ENV:
        print(f"  {key}={os.environ[key]}")


def check_uv() -> None:
    print("Checking if uv installed...")

    uv_path = shutil.which("uv")

    if uv_path is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", "uv"],
            check=True,
        )

        # Refresh PATH for pip --user installs
        user_bin = os.path.expanduser("~/.local/bin")
        os.environ["PATH"] = f"{user_bin}:{os.environ['PATH']}"

        uv_path = shutil.which("uv")

    if uv_path is None:
        raise RuntimeError(
            "Installed uv, but still cannot find uv in system PATH."
        )

    print(f"Found uv installation at {uv_path}")


def setup() -> None:
    configure_uv()

    check_uv()

    print("Syncing venv...")
    subprocess.run(
        ["uv", "sync"],
        check=True,
    )
    print("Sync complete!")

    print("Setting up git hooks...")
    subprocess.run(
        ["git", "config", "core.hooksPath", ".githooks"],
        check=True,
    )
    print("Set up githooks at .githooks")

    print("Setup Complete!")


if __name__ == "__main__":
    setup()