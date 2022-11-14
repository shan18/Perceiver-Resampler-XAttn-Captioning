import argparse
import os

from dataset import MLSLTDataset


def create_dataset(data_root: str, data_type: str):
    """Creates the dataset and its dataloaders

    Args:
        data_root: The root directory of the dataset containing
            the video directory and the json file.
        data_type: The type of the dataset. Can be either `train`, `dev`, or `test`.
    """
    dataset = MLSLTDataset(
        os.path.join(data_root, data_type), os.path.join(data_root, f'{data_type}.json')
    )
    return dataset


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default=os.path.join(BASE_DIR, 'data'),
        help='Path containing the train, dev and test dataset',
    )
    args = parser.parse_args()
