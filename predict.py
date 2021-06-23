# torch
import torch
from torch.utils.data import DataLoader
# others
import getopt
import math
import numpy as np
import os
import PIL
import PIL.Image
import sys
import argparse
# base dir
from dataloader import transforms
from dataloader import dataloader
import model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

assert (int(str('').join(torch.__version__.split('.')[0:2])) >= 13)  # requires at least pytorch version 1.3.0
torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/data_zz/SceneFlow/', type=str, help='Training dataset')
parser.add_argument('--dataset_name', default='SceneFlow', type=str, help='Dataset name')
parser.add_argument('--predict_type', default='train', type=str, help='Dataset type')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size for validation')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loading')
parser.add_argument('--mode', default='test', type=str,
                    help='Validation mode on small subset or test mode on full test data')

args = parser.parse_args()

# nms
def edge_nms(centermaps, kernel_size = 3):
    height, width = centermaps.shape[2:]
    map_nms = torch.nn.functional.max_pool2d(centermaps, kernel_size, 1, 1)
    map_after_nms = (map_nms == centermaps).float() * (centermaps)
    if False:
        map_after_nms_reshaped = map_after_nms.reshape(-1)
        topk_conf, topk_idx = map_after_nms_reshaped.topk(k=self.topk_center)

        topk_x = topk_idx % width
        topk_y = topk_idx // width

        centers_xyv = torch.stack((topk_x.float(), topk_y.float(), topk_conf), dim=-1)
    return map_after_nms

def main():
    # For reproducibility
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train loader
    train_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    train_transform = transforms.Compose(train_transform_list)

    train_data = dataloader.StereoDataset(data_dir=args.data_dir,
                                          dataset_name=args.dataset_name,
                                          mode='train' if args.mode != 'train_all' else 'train_all',
                                          transform=train_transform)

    print('=> {} training samples found in the training set'.format(len(train_data)))

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Validation loader
    val_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    val_transform = transforms.Compose(val_transform_list)
    val_data = dataloader.StereoDataset(data_dir=args.data_dir,
                                        dataset_name=args.dataset_name,
                                        mode=args.mode,
                                        transform=val_transform)

    val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    print('=> {} testing samples found in the testing set'.format(len(val_data)))

    # network
    net = model.Network().to(device)
    if torch.cuda.device_count() > 1:
        print('=> Use %d GPUs' % torch.cuda.device_count())
        net = torch.nn.DataParallel(net)

    # predict
    if args.predict_type == 'test':
        data_loader = val_loader
        dict_name = 'edge_dict_test.npy'
    else:
        data_loader = train_loader
        dict_name = 'edge_dict_train.npy'
    net.eval()
    num_samples = len(data_loader)
    edge_dict = {}
    for i, sample in enumerate(data_loader):
        if i % 100 == 0:
            print('=>predicting %d/%d' % (i, num_samples))
        img = sample['left'].to(device)
        output_nms = edge_nms(net(img)).cpu()
        # import pdb;pdb.set_trace()
        for j in range(output_nms.shape[0]):
            edge_map = output_nms[j, 0]
            edge_points = torch.flip(torch.nonzero(edge_map), [1]).numpy()
            edge_dict[sample['path'][j]] = edge_points

    np.save(dict_name, edge_dict)


if __name__ == "__main__":
    main()


