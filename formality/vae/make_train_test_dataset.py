import click
from dataset import PennTreeBankDataset

@click.command()
@click.option(
    "-i",
    "--dataset-dir",
    default="../data/penn-treebank/",
    show_default=True,
    help="Input Data Directory",
)
def main(dataset_dir):

    train_dataset = PennTreeBankDataset(
        path=dataset_dir, split="train", load_dataset=False,
    )
    valid_dataset = PennTreeBankDataset(
        path=dataset_dir, split="valid", load_dataset=False,
    )
    test_dataset = PennTreeBankDataset(
        path=dataset_dir, split="test", load_dataset=False,
    )

if __name__=='__main__':
    main()
