import torch
from models.cnn_vit import ViT
from datasets.mnist_dataset import get_mnist_dataloaders
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import utils as my_utils


def compute_loss_function(pred, label):
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(pred, label)
    return loss

def evaluate_model(test_loader, model, patch_size, device="cpu"):
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
    my_utils.save_model(model, model_dir="../artifacts", patch_size=patch_size)
    model.train()


def train():
    batch_size = 128
    num_epochs = 5
    patch_size = 4
    train_loader, test_loader = get_mnist_dataloaders(batch_size)
    device = my_utils.get_device()
    model = ViT(
        img_size=(28, 28, 1),
        patch_size=patch_size,
        embed_size=128,
        num_classes=10,
    ).to(device)
    my_utils.save_model(model, model_dir="../artifacts", patch_size=patch_size)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        model.train()
        with tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                x, label = batch
                label = label.to(device)
                x = x.to(device)
                y = model(x)
                loss = compute_loss_function(y, label)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"loss": loss.item()})
        evaluate_model(test_loader, model, device=device, patch_size=patch_size)


if __name__ == "__main__":
    train()
