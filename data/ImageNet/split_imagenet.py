import json
import sys
import argparse
import numpy as np
import os
import shutil


def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(
        description='Determines the random seed based on the input argument.')

    parser.add_argument('--seed', type=int, default=6,
                        help='Random seed')
    parser.add_argument('--num-train-classes', type=int, default=800,
                        help='Number of classes in the training set')
    parser.add_argument('--num-val-classes', type=int, default=200,
                        help='Number of classes in the validation set')
    # parser.add_argument('--num-test-classes', type=int, default=0,
    #                     help='Number of classes in the test set')
    parser.add_argument('--not-backup-dataset', action='store_true',
                        help='Whether to backup the original dataset')

    args = parser.parse_args(args)
    return args


def main(args):
    """
    Main function.
    """
    args = parse_args(args)
    print(json.dumps(vars(args), indent=2))

    np.random.seed(args.seed)

    if args.not_backup_dataset:
        pass
    else:
        if not os.path.exists('./train_backup'):
            shutil.copytree('./train', './train_backup')
        if not os.path.exists('./val_backup'):
            shutil.copytree('./val', './val_backup')

    all_classes = [
        item for item in os.listdir('./train')
        if os.path.isdir(os.path.join('./train', item))
    ]
    assert len(all_classes) == 1000, 'There should be 1000 classes in total.'

    num_train_and_val_classes = args.num_train_classes + args.num_val_classes
    assert num_train_and_val_classes == 1000, 'The number of classes in the training and validation set should sum to 1000.'

    shuffled_all_classes = np.random.permutation(all_classes)
    train_classes = shuffled_all_classes[:args.num_train_classes]
    val_classes = shuffled_all_classes[-args.num_val_classes:]

    masked_train_dir = './masked_train'
    if not os.path.exists(masked_train_dir):
        os.mkdir(masked_train_dir)
    zero_shot_dir = './zero_shot'
    if not os.path.exists(zero_shot_dir):
        os.mkdir(zero_shot_dir)

    for val_class in val_classes:
        os.rename('./train/' + val_class, './masked_train/' + val_class)
        os.rename('./val/' + val_class, './zero_shot/' + val_class)

    # Check if the remaining subdirectories in the train directory match train_classes
    remaining_train_classes = [
        item for item in os.listdir('./train')
        if os.path.isdir(os.path.join('./train', item))
    ]
    assert set(remaining_train_classes) == set(
        train_classes), 'The remaining subdirectories in the train directory do not match train_classes.'

    # Check if the remaining subdirectories in the val directory match train_classes
    remaining_val_classes = [
        item for item in os.listdir('./val')
        if os.path.isdir(os.path.join('./val', item))
    ]
    assert set(remaining_val_classes) == set(
        train_classes), 'The remaining subdirectories in the val directory do not match train_classes.'

    # Check if the subdirectories in the masked_train directory match val_classes
    masked_train_classes = [
        item for item in os.listdir('./masked_train')
        if os.path.isdir(os.path.join('./masked_train', item))
    ]
    assert set(masked_train_classes) == set(
        val_classes), 'The subdirectories in the masked_train directory do not match val_classes.'

    # Check if the subdirectories in the zero_shot directory match val_classes
    zero_shot_classes = [
        item for item in os.listdir('./zero_shot')
        if os.path.isdir(os.path.join('./zero_shot', item))
    ]
    assert set(zero_shot_classes) == set(
        val_classes), 'The subdirectories in the zero_shot directory do not match val_classes.'

    print('Successfully split the ImageNet dataset into training and validation sets.')


if __name__ == '__main__':
    main(sys.argv[1:])
