import torch
from tqdm import tqdm
from utils.trainer_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_model()
            if epoch % self.config["OPTIM"]["save_ckpt_every"] == 0:
                self.save_checkpoint()

    def train_one_epoch(self):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for n_iter, (imgs, labels) in pbar:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(imgs)
            loss, loss_dict = self.loss_fn(preds, labels)
            self.write_dict_to_tb(loss_dict, self.total_iters, prefix="train")

            loss.backward()
            self.gradient_clip()
            self.accumulate_gradients()

            self.total_iters += 1
            pbar.set_postfix(
                {
                    "mode": "train",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                }
            )
        pbar.close()

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        for n_iter, (imgs, labels) in pbar:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(imgs)
            loss, loss_dict = self.loss_fn(preds, labels)
            self.write_dict_to_tb(loss_dict, self.total_iters, prefix="validation")

            self.total_iters += 1
            pbar.set_postfix(
                {
                    "mode": "validation",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                }
            )
        pbar.close()