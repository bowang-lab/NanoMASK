import numpy as np
import pandas as pd
import os
join = os.path.join
import nibabel as nib
import medpy.metric.binary
from collections import OrderedDict
import argparse


parser = argparse.ArgumentParser(description='Program to compute the DSC and VD of the segmentation per organ and output as a csv file')
parser.add_argument('-s', '--segmentation', help='The path of segmentation folder', required=True)
parser.add_argument('-r', '--reference', help='The path of ground truth folder', required=True)
parser.add_argument('-o', '--output', help='The path of output csv folder', required=True)
parser.add_argument('-n', '--names', nargs='+', help='List of organs in ascending order by the mapped integer', required=True)
args = parser.parse_args()


def remove_DS_Store(path):
    if os.path.exists(join(path, '.DS_Store')):
        os.remove(join(path, '.DS_Store'))

result_path = args.segmentation
refer_path = args.reference

remove_DS_Store(result_path)
remove_DS_Store(refer_path)


organ_dict = {}
num = 1

for organ in args.names:
    organ_dict[organ] = num
    num += 1

label_tolerance = OrderedDict(organ_dict)


seg_metrics = OrderedDict()  
seg_metrics['Names'] = []                 
for organ in label_tolerance.keys():
    seg_metrics['{}_DSC'.format(organ)] = list()
for organ in label_tolerance.keys():
    seg_metrics['{}_VD'.format(organ)] = list()

results = sorted(os.listdir(result_path))

for i in results:
    seg_metrics['Names'].append(i)
    result_nii = nib.load(join(result_path, i))
    result_arr = result_nii.get_fdata()

    refer_nii = nib.load(join(refer_path, i))
    refer_arr = refer_nii.get_fdata()

    for index, organ in enumerate(label_tolerance.keys(), 1):
        organ_arr_result = np.where(result_arr == index, 1, 0)
        organ_arr_refer = np.where(refer_arr == index, 1, 0)

        if np.all(organ_arr_refer == 0) and np.all(organ_arr_result == 0):
            dice_score = 1
            vd = 0

        elif np.all(organ_arr_refer == 0) and not np.all(organ_arr_result == 0):
            print(i + " " + organ)
            dice_score = 0
            vd = 1 

        else:
            dice_score = medpy.metric.binary.dc(organ_arr_result, organ_arr_refer)
            volume_seg = np.sum(organ_arr_result)
            volume_gt = np.sum(organ_arr_refer)
            vd = abs((volume_seg - volume_gt) / volume_gt)

        seg_metrics['{}_DSC'.format(organ)].append(round(dice_score, 4))
        seg_metrics['{}_VD'.format(organ)].append(round(vd, 4))


dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv(args.output, index=False)