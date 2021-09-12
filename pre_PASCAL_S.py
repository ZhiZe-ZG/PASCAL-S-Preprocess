'''
SIR preprocess script
used to generate pre processed data for SIR
simple trans origin data into torch tensor files,
not generate graph
process PASCAL-S dataset
usage:
python pre_PASCAL_S.py -r './salObj' -w './SIR_PASCAL_S'
for each output file:
    image: means origin image
    saliency: means PASCAL-S saliency map
    full_seg: means PASCAL-S full segmentation
    fixation: means PASCAL-S fixation data
'''

from typing import Any
import os
import argparse
# import numpy as np
import skimage.io as io
# import networkx as nx
# import random
import torch
import tables

import sys

print(os.getcwd())  # bash pwd
print(sys.path)  # file path
sys.path.append(os.path.join(sys.path[0], '..'))
# from SLIC_Graph import ImageToGraph
# from ZhiZeUtils import *

# command line arguments parser
parser = argparse.ArgumentParser(
    description='used to generate pre processed data for SIR')

# add parse arguments
parser.add_argument('-r',
                    '--readpath',
                    metavar='path',
                    type=str,
                    help='the path to read',
                    required=True)
parser.add_argument('-w',
                    '--writepath',
                    metavar='path',
                    type=str,
                    help='the path to write',
                    required=True)

args = parser.parse_args()
config = vars(args)

# check read path and write path
if not os.path.exists(config['readpath']):
    print('Wrong read path!')
    exit(1)
if not os.path.exists(config['writepath']):
    try:
        os.makedirs(config['writepath'])
    except Exception as e:
        print(e)
        print('Wrong write path and can not create it.')
        exit(1)

# config read paths
TR_read_image_path = os.path.join(config['readpath'], 'datasets', 'imgs',
                                  'pascal')
TR_read_mask_path = os.path.join(config['readpath'], 'datasets', 'masks',
                                 'pascal')
full_seg_path = os.path.join(config['readpath'], 'datasets', 'segments',
                             'pascal', 'isoCell.mat')
fixation_path = os.path.join(config['readpath'], 'datasets', 'fixations',
                             'pascalFix.mat')

# total write path
TR_write_path = os.path.join(config['writepath'])

# check write path
if not os.path.exists(TR_write_path):
    try:
        os.makedirs(TR_write_path)
    except Exception as e:
        print(e)
        print('Wrong write path and can not create it.')
        exit(1)

# set suffix
image_suffix = '.jpg'
mask_suffix = '.png'
save_suffix = '.pt'


# get name list
def get_name_list(path):
    name_list = os.listdir(path)
    # get file name suffix
    suffix = os.path.splitext(name_list[0])[-1]
    # remove all suffix
    name_list = [
        n.replace(suffix, '') for n in name_list if n.endswith(suffix)
    ]
    return name_list


TR_names = get_name_list(TR_read_image_path)


# check data error
def check_size_match(image_root, mask_root, name):
    image_path = os.path.join(image_root, name + image_suffix)
    mask_path = os.path.join(mask_root, name + mask_suffix)
    image = io.imread(image_path)
    mask = io.imread(mask_path)
    return image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1]


TR_names = filter(
    lambda n: check_size_match(TR_read_image_path, TR_read_mask_path, n),
    TR_names)

print('Chicking files...')
TR_names = [n for n in TR_names]
TR_len = len(TR_names)
print('total data:', TR_len)


# read file
def create_reader(image_root, mask_root, full_segs, fixations):
    def reader(name):
        image_path = os.path.join(image_root, name + image_suffix)
        mask_path = os.path.join(mask_root, name + mask_suffix)
        image = io.imread(image_path)
        mask = io.imread(mask_path)
        idx_num = ((int)(name))
        full_seg = full_segs[idx_num - 1]
        fixation = fixations[idx_num - 1]
        # translate to torch Tensor
        image = torch.Tensor(image)
        mask = torch.Tensor(mask)
        full_seg = torch.Tensor(full_seg).squeeze()
        fixation = torch.Tensor(fixation).squeeze()
        return name, image, mask, full_seg, fixation

    return reader


# load full segmentation mat file
full_seg_file = tables.open_file(full_seg_path)
full_segs = full_seg_file.root.isoCell[0]
# load fixation mat file
fixation_file = tables.open_file(fixation_path)
fixations = fixation_file.root.fixCell[0]
# use map, not list, save some memory
TR_reader = create_reader(TR_read_image_path, TR_read_mask_path, full_segs,
                          fixations)
TR_stream = map(lambda n: TR_reader(n), TR_names)


def save_file(name: str, image: Any, mask: Any, full_seg: Any,
              fixation: Any) -> None:
    print(f'processing... {name} ')
    data_dict = {
        'image': image,
        'saliency': mask,
        'full_seg': full_seg,
        'fixation': fixation
    }
    save_path = os.path.join(TR_write_path, name + save_suffix)
    torch.save(data_dict, save_path)
    return


TR_stream = map(lambda n: save_file(*n), TR_stream)

_ = [n for n in TR_stream]

print('Preprocess done.')
