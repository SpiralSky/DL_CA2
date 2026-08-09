from pathlib import Path

import torch


def main(
    weights_dir: Path = Path("weights"),
    build_dir: Path = Path("build"),
) -> None:
    print(f"Building weights from {weights_dir.resolve()}")

    build_dir.mkdir(exist_ok=True)

    checkpoints = [file for file in weights_dir.iterdir() if file.is_file()]

    if not checkpoints:
        print("No checkpoint files found.")
        return

    for checkpoint in checkpoints:
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)

        if "model" not in data:
            print(f"Skipping {checkpoint.name}: no model weights")
            continue

        output = build_dir / checkpoint.name.replace(".pt", ".pth")

        torch.save(data["model"], output)

        print(f"Created {output}")


if __name__ == "__main__":
    main()