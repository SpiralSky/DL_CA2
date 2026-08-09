import shutil
import subprocess
import sys

def check_uv() -> None:
    print("Checking if uv installed...")
    # noinspection deprecation
    uv_path = shutil.which("uv")
    if uv_path is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", "uv"],
            check=True
        )

        # noinspection deprecation
        uv_path = shutil.which("uv")

        if uv_path is None:
            raise RuntimeError(
                "Installed uv, but still cannot find uv in system PATH."
            )

        print(f"Successfully installed uv at {uv_path}")
    else:
        print(f"Found uv installation at {uv_path}")

def setup() -> None:
    check_uv()

    print("Syncing venv...")
    subprocess.run(
        ["uv", "sync"],
        check=True
    )
    print("Sync complete!")

    print("Setting up git hooks...")
    subprocess.run(
        ["git", "config", "core.hooksPath", ".githooks"],
        check=True
    )
    print("Set up githooks at .githooks")
    print("Setup Complete!")

if __name__ == "__main__":
    setup()