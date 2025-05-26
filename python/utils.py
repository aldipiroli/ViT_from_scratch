import torch
from pathlib import Path


def get_device():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    return device


def save_model(model, model_dir, patch_size):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_dir) / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved Model in: {model_path}")

    model.eval()
    example_input = torch.randn(1, 1, 28, 28).to(get_device())

    traced_script_module = torch.jit.trace(model, example_input)
    model_path = Path(model_dir) / "model_ts.pt"
    traced_script_module.save(model_path)
    print(f"Saved TorchScript Model in: {model_path}")

def divide_in_patches(z, patch_size):
    all_patches = []
    b, h, w, c = z.shape
    for i in range(0, int(w / patch_size)):
        for j in range(0, int(h / patch_size)):
            curr_patch = z[
                :,
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
                :,
            ]
            curr_patch = curr_patch.reshape(b, patch_size * patch_size * c)
            all_patches.append(curr_patch)
    all_patches = torch.stack(all_patches, 1)  # bxNx(patch_size**2xc)
    return all_patches