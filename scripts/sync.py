import subprocess
import sys

def sync(silent=False, start_message="Syncing notebooks...", end_message="Sync complete!"):
    """Sync all paired notebooks in the project."""
    print(start_message)
    result = subprocess.run([sys.executable, "-m", "jupytext", "--sync", "./notebooks/*.py"], check=True, capture_output=True, text=True)
    output = "\n".join(line for line in result.stdout.splitlines() if "Warning" not in line)
    if not silent:
        print(output)
    print(end_message)