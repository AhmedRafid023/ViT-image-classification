import lightning as L
import yaml
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset_imagenet import TinyImageNetDataModule
from model_imagenet import ProjectorAblationModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_id = config['model']['id'].split("/")[-1]
    proj = config['model']['projector_type']
    sit = config['experiment']['situation']
    
    print(f"Training: {model_id} | Proj: {proj} | Sit: {sit}")

    dm = TinyImageNetDataModule(config)
    model = ProjectorAblationModel(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{model_id}_{proj}_{sit}",
        filename=f"{config['experiment']['situation']}-{{epoch:02d}}",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        save_last = True
    )

    trainer = L.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        precision=config['trainer']['precision'], 
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()