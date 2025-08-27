import argparse

from dataset.mnist_dataset import MNISTDataset
from models.vit import SimpleViT
from models.cls_loss import ClsLoss
from utils.misc import get_logger, load_config, make_artifacts_dirs
from utils.trainer import Trainer


def train(args):
    config = load_config(args.config)
    config = make_artifacts_dirs(config, log_datetime=True)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    train_dataset = MNISTDataset(cfg=config, mode="train")
    val_dataset = MNISTDataset(cfg=config, mode="val")

    model = SimpleViT(config)
    trainer.set_model(model)

    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(ClsLoss())
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/vit_config.yaml", help="Config path")
    args = parser.parse_args()
    train(args)
