# Step 0: Add the libraries and argument parser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from collections import OrderedDict
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import utils
import vision_transformer as vits
from tifffile import imread
import albumentations as A
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from utils import cosine_scheduler
import time
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('DINO CLASSIFICATION', add_help=False)

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="path to the pretrained model.")
    parser.add_argument('--checkpoint_key', default='teacher', type=str, required=True,
                        help="read student or teacher model from the pretrained model.")
    parser.add_argument('--arch', default='vit_base', type=str,
                        help="""name of architecture to train. For quick experiments with ViTs,
                            we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int,
                        help="patch size in pixels.")
    parser.add_argument('--num_classes', default=16, type=int,
                        help="number of classes in the dataset.")
    parser.add_argument("--lr", default=0.0001, type=float,
                        help="learning rate")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument("--world_size", default=1, type=int, help="world size.")
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--data_path', type=str, required=True,
                        help="path to the data.")
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='proportion of the dataset to be used for training. rest is for validation.')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='number of data loading workers per GPU.')
    parser.add_argument('--output_dir', default=".", type=str,
                        help='path to save logs and checkpoints.')
    parser.add_argument('--max_epochs', default=10, type=int,
                        help='number of training epochs.')
    parser.add_argument('--resume', type=str,
                        help="path to checkpoint to resume training.")
    return parser


def img_loader(path):
    return imread(path)


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


# basically the same as MAE transform
class DINODownstreamTransform:
    def __init__(self, input_size):
        self.input_size = input_size

        self.transforms = A.Compose([
            A.RandomCrop(height=self.input_size,
                         width=self.input_size,
                         always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=(0.2, 0.3),
                                       contrast_limit=(0.2, 0.3),
                                       p=0.2),
            A.RandomGamma(gamma_limit=(100, 140), p=0.2),
            A.RandomToneCurve(scale=0.1, p=0.2)
        ])

    def __call__(self, image):
        # Make sure image is a np array
        if type(image) == 'torch.Tensor':
            image = image.numpy()

        crop = ToTensor()(self.transforms(image=image)['image'])
        return crop


class DINOClassification(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.automatic_optimization = False

        self.args = args
        checkpoint = self.load_checkpoint(self.args.checkpoint_path, self.args.checkpoint_key)
        self.backbone = self.prepare_arch(self.args.arch, checkpoint, self.args.patch_size)
        self.backbone.eval()
        self.linear_classifier = LinearClassifier(self.backbone.embed_dim, self.args.num_classes)

        # self.model = torch.nn.Sequential(model_vit, linear_classifier)
        self.train_dataset, self.val_dataset = self.build_dataset(self.args)
        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)

        self.lr = self.args.lr * (self.args.batch_size_per_gpu * self.args.world_size / 256)
        # self.lr_sch = cosine_scheduler(self.lr, 1e-6, self.args.max_epochs, )
        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def load_checkpoint(checkpoint_path, checkpoint_key):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint = checkpoint['state_dict']
        pretrained_model = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith(checkpoint_key):
                pretrained_model[k] = v

        pretrained_model = {k.replace(f"{checkpoint_key}.", ""): v for k, v in pretrained_model.items()}
        pretrained_model = {k.replace("backbone.", ""): v for k, v in pretrained_model.items()}
        return pretrained_model

    @staticmethod
    def prepare_arch(arch, pretrained_model, patch_size):
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        msg = model.load_state_dict(pretrained_model, strict=False)
        return model

    @staticmethod
    def build_dataset(args):
        transform = DINODownstreamTransform(args.input_size)
        dataset = ImageFolder(args.data_path, transform=transform, loader=img_loader)
        train_dataset, val_dataset = random_split(dataset, [args.train_ratio, 1 - args.train_ratio])
        return train_dataset, val_dataset

    def configure_optimizers(self):
        regularized, not_regularized = [], []
        for n, p in self.linear_classifier.named_parameters():
            if not p.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if n.endswith(".bias") or len(p.shape) == 1:
                not_regularized.append(p)
            else:
                regularized.append(p)
        param_groups = [{'params': regularized},
                        {'params': not_regularized, 'weight_decay': 0.}]  # weight decay is 0 because of the scheduler
        opt = torch.optim.AdamW(param_groups, self.lr)
        # scheduler = CosineAnnealingWarmRestarts(optimizer=opt,
        #                                         T_0=100,
        #                                         T_mult=3,
        #                                         eta_min=1e-6)  # eta_min=1e-6)
        # lr_scheduler = {
        #     'scheduler': scheduler,
        #     'name': 'learning_rate',
        #     'interval': 'step',
        #     'frequency': 1
        # }
        # return [opt], [lr_scheduler]
        return opt

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_dataset,
                                  sampler=self.train_sampler,
                                  batch_size=self.args.batch_size_per_gpu,
                                  num_workers=self.args.num_workers,
                                  pin_memory=True,
                                  drop_last=True,)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(dataset=self.val_dataset,
                                sampler=self.val_sampler,
                                batch_size=self.args.batch_size_per_gpu,
                                num_workers=self.args.num_workers,
                                pin_memory=True,
                                drop_last=False,)
        return val_loader

    def training_step(self, batch):
        opt = self.optimizers()
        # lr_sch = self.lr_schedulers()
        # lr_sch.step()

        samples = batch[0]
        targets = batch[1]
        output_backbone = self.backbone(samples)
        output = self.linear_classifier(output_backbone)
        loss = self.loss(output, targets)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        # lr_sch.step(self.current_epoch)
        # self.log('train/lr', self.lr)
        self.log('train/loss', loss)

    def validation_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1]

        output_backbone = self.backbone(samples)
        output = self.linear_classifier(output_backbone)
        loss = self.loss(output, targets)

        acc1, acc5 = utils.accuracy(output, targets, topk=(1, 5))
        self.log('test/loss', loss, True)
        self.log('test/acc1', acc1, True)
        self.log('test/acc5', acc5, True)


def main(args):
    dino_class = DINOClassification(args)
    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,
                                          every_n_epochs=int(args.max_epochs/3),
                                          # every_n_train_steps=int(args.max_steps / 5),
                                          save_last=True)
    lr_callback = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger(save_dir=args.output_dir,
                               name="",
                               default_hp_metric=False)

    trainer = L.Trainer(accelerator='gpu',
                        max_epochs=args.max_epochs,
                        default_root_dir=args.output_dir,
                        enable_progress_bar=True,
                        logger=logger,
                        log_every_n_steps=10,
                        callbacks=[checkpoint_callback, lr_callback])

    print("beginning the training.")
    start = time.time()
    if args.resume:
        trainer.fit(model=dino_class, ckpt_path=args.resume)
    else:
        trainer.fit(model=dino_class)
    # trainer.fit(model=dino)
    end = time.time()
    print(f"training completed. Elapsed time {end - start} seconds.")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)


