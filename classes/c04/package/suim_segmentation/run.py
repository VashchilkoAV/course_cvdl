
import argparse
from pathlib import Path
import wandb

import torch
from torch.utils import data as tdata
from tqdm import tqdm

from .data import SuimDataset, EveryNthFilterSampler
from .model import SuimModel
from .loss import DiceLoss
from .metrics import Accuracy
from .trainer import Trainer


def run_pipeline(args):
    wandb.init(project='test1', config=vars(args))
    device = args.device
    model = SuimModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_val_ds = SuimDataset(Path(args.train_data), masks_as_color=False, target_size=(256, 256))
    test_ds = SuimDataset(Path(args.test_data), masks_as_color=False, target_size=(256, 256))
    
    test_iter = tdata.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    train_iter = tdata.DataLoader(train_val_ds, batch_size=args.batch_size, 
        sampler=EveryNthFilterSampler(dataset_size=len(train_val_ds), n=5, pass_every_nth=False, shuffle=True)
    )
    val_iter = tdata.DataLoader(train_val_ds, batch_size=args.batch_size, 
        sampler=EveryNthFilterSampler(dataset_size=len(train_val_ds), n=5, pass_every_nth=True, shuffle=False)
    )
    
    loss = DiceLoss()
    metric = Accuracy()
    
    trainer = Trainer(
        net=model,
        opt=opt,
        train_loader=train_iter,
        val_loader=val_iter,
        test_loader=test_iter,
        loss=loss,
        metric=metric,
    )
    
    mean = lambda values: sum(values) / len(values)
    
    for e in range(args.num_epochs):
        print(f"Epoch {e}")
        with_testing = (e == args.num_epochs - 1)
        epoch_stats = trainer(num_epochs=1, with_testing=with_testing)
        train_loss, train_metric = epoch_stats['train'][0]
        val_loss, val_metric = epoch_stats['val'][0]
        wandb.log({"train": {"loss": train_loss, "metric": train_metric}, 
                   "val": {"loss": val_loss, "acc": val_metric}}, step=e)
        assert isinstance(train_loss, list), type(train_loss)
    
    if args.num_epochs > 0:
        test_loss, test_metric = epoch_stats['test'][0]
        wandb.run.summary['test.loss'] = mean(test_loss)
        wandb.run.summary['test.metric'] = mean(test_metric)

    wandb.run.summary['haha'] = 'hehe'
    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default='cpu:0')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
    print("Finished")
