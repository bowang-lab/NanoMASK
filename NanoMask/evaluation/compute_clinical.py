import numpy as np
import pandas as pd
import os
join = os.path.join
import nibabel as nib
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser(description='Program to compute the clinical metrics per organ for different experiments, mice, and timepoints, and output as an xlsx file')
parser.add_argument('-s', '--segmentation', help='The path of segmentation folder', required=True)
parser.add_argument('-r', '--reference', help='The path of ground truth folder', required=True)
parser.add_argument('-i', '--image', help='The path of PET/CT data folder', required=True)
parser.add_argument('-o', '--output', help='The path of output xlsx folder', required=True)
parser.add_argument('-m', '--machine', help='The path of machine file with % injected dose information', required=True)
parser.add_argument('-n', '--names', nargs='+', help='List of organs in ascending order by the mapped integer', required=True)
args = parser.parse_args()


def remove_DS_Store(path):
    if os.path.exists(join(path, '.DS_Store')):
        os.remove(join(path, '.DS_Store'))

img_path = args.image
gt_path = args.regerence
seg_path = args.segmentation
organs = args.names

organ_dict = {}
num = 1

for organ in args.names:
    organ_dict[organ] = num
    num += 1

label_tolerance = OrderedDict(organ_dict)


clinical_measures = OrderedDict() 
clinical_measures['Experiment'] = list()
clinical_measures['Mouse #'] = list() 
clinical_measures['Timepoint'] = list()
clinical_measures['Organ'] = list() 

for column in ['MeanVol', 'RegionVol', 'Intensity', 'Max', 'Min', 'Std']:
    clinical_measures['GT-' + column] = list()
    clinical_measures['Seg-' + column] = list()        

remove_DS_Store(img_path)
remove_DS_Store(gt_path)
remove_DS_Store(seg_path)

for organ in organs:
    for contour in sorted(os.listdir(seg_path)):
        name = contour.split('.nii.gz')[0]
        pet_name = name + '_0001.nii.gz'

        # all CT/PET data follow the name convention: experiment_mouse_timepoint_0000.nii.gz (CT) and experiment_mouse_timepoint_0001.nii.gz (PET)
        lst = name.split('_')
        if len(lst) == 5:
            experiment = lst[0] + '_' + lst[1] + '_' + lst[2]
            mouse = lst[3]
            tp = lst[4]
        elif len(lst) == 4:
            experiment = lst[0] + '_' + lst[1]
            mouse = lst[2]
            tp = lst[3]

        clinical_measures['Experiment'].append(experiment)
        clinical_measures['Mouse #'].append(mouse)
        clinical_measures['Timepoint'].append(tp)
        clinical_measures['Organ'].append(organ)

        pet_nii = nib.load(join(img_path, pet_name))
        pet = pet_nii.get_fdata()

        for path in [gt_path, seg_path]:
            mask = nib.load(join(path, contour)).get_fdata()
            i = label_tolerance[organ]

            mean_voxel = np.mean(pet[mask==i])
            total_volume = np.sum(mask==i) * np.prod(pet_nii.header.get_zooms())
            total_intensity = np.sum(pet[mask==i])

            if pet[mask==i].size > 0:
                max = np.max(pet[mask==i])
                min = np.min(pet[mask==i])
                std = np.std(pet[mask==i])

            # contour not exist
            else:
                max = ''
                min = ''
                std = ''

            if 'labelsTr' in path:
                clinical_measures['GT-MeanVol'].append(mean_voxel*0.0272)
                clinical_measures['GT-RegionVol'].append(total_volume)
                clinical_measures['GT-Intensity'].append(total_intensity*0.0298)
                clinical_measures['GT-Max'].append(max)
                clinical_measures['GT-Min'].append(min)
                clinical_measures['GT-Std'].append(std)
            else:
                clinical_measures['Seg-MeanVol'].append(mean_voxel*0.0272)
                clinical_measures['Seg-RegionVol'].append(total_volume)
                clinical_measures['Seg-Intensity'].append(total_intensity*0.0298)
                clinical_measures['Seg-Max'].append(max)
                clinical_measures['Seg-Min'].append(min)
                clinical_measures['Seg-Std'].append(std)

dataframe = pd.DataFrame(clinical_measures)


# merge gt and seg with machine

df_ori = pd.read_excel(args.machine)

dataframe['Mouse #'] = dataframe['Mouse #'].str.replace(' ', '')  # Extra 1 -> Extra1

df_ori = df_ori.iloc[1:, :]
df_ori.rename(columns={df_ori.columns[0]: 'Experiment', df_ori.columns[2]: 'Timepoint'}, inplace=True)

# remove single quote
for i in range(4):
    df_ori[df_ori.columns[i]] = df_ori[df_ori.columns[i]].str.strip("'")

# remove x in front of the experiment
for element in df_ori[df_ori.columns[0]]:
    if 'x' in element:
        df_ori[df_ori.columns[0]] = df_ori[df_ori.columns[0]].replace([element], element[1:])

# get the timepoint by removing Contour_
df_ori[df_ori.columns[2]] = df_ori[df_ori.columns[2]].str.split('_').str[1]

# remove organ that is Left or Right
df1 = df_ori[(df_ori.iloc[:, 3] != "Left") | (df_ori.iloc[:, 3] != "Right")]

# keep necessary columns, include fields needed to calculate idpercc
df1 = df1[[df1.columns[0], df1.columns[1], df1.columns[2], df1.columns[3], 
df1.columns[5], df1.columns[7], df1.columns[8], df1.columns[20], df1.columns[17], df1.columns[18]]]
# scantime, injtime, dose, interval

df1.rename(columns={'Mean Voxel Intensity (nCi/cc)': 'Machine-MeanVol', 'Total Region Volume (mm^3)': 'Machine-RegionVol', 
'Total Region Intensity (nCi/cc)': 'Machine-Intensity', '% Injected Dose / cc': 'Machine-IDpercc', 'Injected Dose (mCi)': 'dose',
'Time Interval Between Injection and Scan (datetime format, units of days)': 'int'}, inplace=True)

new_df = pd.merge(dataframe, df1, how='inner', 
left_on=['Experiment', 'Mouse #', 'Timepoint', 'Organ'], right_on = [df1.columns[0], df1.columns[1], df1.columns[2], df1.columns[3]])

gt = new_df['GT-MeanVol']
seg = new_df['Seg-MeanVol']
dose = new_df['dose']
diff = new_df['int']
new_df['GT-IDpercc'] = (gt / 10**6 / (2**(-diff/0.5291666)))/dose * 100
new_df['Seg-IDpercc'] = (seg / 10**6 / (2**(-diff/0.5291666)))/dose * 100

# reorder columns
new_df = new_df[['Experiment', 'Mouse #', 'Timepoint', 'Organ', 'Machine-MeanVol', 'GT-MeanVol', 'Seg-MeanVol', 'Machine-RegionVol', 'GT-RegionVol', 'Seg-RegionVol', 
'Machine-Intensity', 'GT-Intensity', 'Seg-Intensity', 'Machine-IDpercc', 'GT-IDpercc', 'Seg-IDpercc', 'GT-Max', 'Seg-Max', 'GT-Min', 'Seg-Min', 'GT-Std', 'Seg-Std']]

new_df.to_excel(args.output, index=False)
