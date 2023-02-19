import os
import numpy as np
import shutil
join = os.path.join

parser = argparse.ArgumentParser(description='Program to use affine registration to register CT and PET data')
parser.add_argument('-i', '--input', help='The input path contains PET/CT data and organ contours', required=True)
parser.add_argument('-m', '--moving', help='The moving image. Value can only be CT or PET. If it is CT, then moving image is CT and fixed image is PET, and vice-versa', required=True)
args = parser.parse_args()

root_path = args.input

def remove_DS_Store(path):
    if os.path.exists(join(path, '.DS_Store')):
        os.remove(join(path, '.DS_Store'))

remove_DS_Store(root_path)
os.chdir(root_path)

# register CT data and organ contours to PET
if args.moving == 'CT':
    mv_img = './CT_0000.nii.gz'
    fix_img = './PET_0001.nii.gz'
    mv_contour = "./organs-CT.nii.gz"
    trans_name = './CT2PET.mat'
    save_img_name = 'reg_CT_0000.nii.gz'
    save_contour_name = 'organs-PET.nii.gz'

# register PET data and organ contours to CT
elif args.moving == 'PET':
    mv_img = './PET_0001.nii.gz'
    fix_img = './CT_0000.nii.gz'
    mv_contour = "./organs-PET.nii.gz"
    trans_name = './PET2CT.mat'
    save_img_name = 'reg_PET_0001.nii.gz'
    save_contour_name = 'organs-CT.nii.gz'

# generate and save the affine transformation
generate_transform_cmd = 'greedy -d 3 -a -m NMI -i {} {} -o {} -ia-image-centers -dof 6 -n 20x20x20x20'.format(fix_img, mv_img, trans_name)
os.system(generate_transform_cmd)

# apply previously generated affine transformation on PET/CT data and organ contours
apply_transform_cmd = 'greedy -d 3 -rf {} -rm {} {} -r {}'.format(fix_img, mv_img, save_img_name, trans_name)
apply_transform_cmd = 'greedy -d 3 -rf {} -ri LABEL 0.2vox -rm {} {} -r {}'.format(fix_img, mv_contour, save_contour_name, trans_name)
os.system(apply_transform_cmd)