import torch
from models.cnn_vit import ViT
from datasets.mnist_dataset import get_mnist_dataloaders
from tqdm import tqdm
import utils as my_utils


def evaluate_model(test_loader, model, patch_size, device="cpu", save_model=True):
    print("Running Evaluation..")
    all_samples = 0
    correct_samples = 0
    model.eval()

    for batch in tqdm(test_loader, desc="Evaluating..", total=len(test_loader)):
        x, label = batch
        x = x.to(device)
        label = label.to(device)
        y = model(x)
        y_pred_sm = torch.nn.functional.softmax(y, -1)
        y_pred = torch.argmax(y_pred_sm, -1)
        for y, l in zip(y_pred, label):
            if y == l:
                correct_samples += 1
            all_samples += 1
    print(
        f"Evaluation results: {correct_samples}/{all_samples}, Accuracy: {(correct_samples/all_samples)*100:.2f}%"
    )
    if save_model:
        my_utils.save_model(model, model_dir="../artifacts", patch_size=patch_size)
    model.train()


def test():
    batch_size = 128
    patch_size = 4
    _, test_loader = get_mnist_dataloaders(batch_size)
    device = my_utils.get_device()
    model = ViT(
        img_size=(28, 28, 1),
        patch_size=patch_size,
        embed_size=128,
        num_classes=10,
    ).to(device)
    model.load_state_dict(torch.load("../artifacts/model.pt", weights_only=True))
    evaluate_model(
        test_loader, model, device=device, patch_size=patch_size, save_model=False
    )


if __name__ == "__main__":
    test()
