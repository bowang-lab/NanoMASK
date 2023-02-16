import os
join = os.path.join
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
import glob
import SimpleITK as sitk
import glob
import argparse


parser = argparse.ArgumentParser(description='Program to convert DICOM into NIfTI')
parser.add_argument('-i', '--input', help='The input root path contains all experiments', required=True)
parser.add_argument('-o', '--output', help='The output path', required=True)
parser.add_argument('-a', '--annotations', help='Convert the organ contours according to CT or PET image. Value can only be CT or PET', required=True)
args = parser.parse_args()

root_path = args.input
save_root_path = args.output


def remove_DS_Store(path):
    if os.path.exists(join(path, '.DS_Store')):
        os.remove(join(path, '.DS_Store'))

def dcm2nii(dcm_path):
    DicomReader = sitk.ImageSeriesReader()
    patient_files = DicomReader.GetGDCMSeriesFileNames(dcm_path)
    DicomReader.SetFileNames(patient_files)
    
    img_sitk = DicomReader.Execute()
    return img_sitk

def convert_contours(contour_path, save_path, img_dcms_path):
    organ_rts = glob.glob(join(contour_path, '*.dcm'))
    if len(organ_rts)>0:
        for idx, rt in enumerate(organ_rts):
            if len(list_rt_structs(rt))>1:
                # organ save nii path
                if idx==0:
                    organ_nii_path = save_path
                else:
                    organ_nii_path = join(save_path, str(idx))
                os.makedirs(organ_nii_path, exist_ok=True)
                # print(organ, list_rt_structs(rt))
                dcmrtstruct2nii(rt, img_dcms_path, organ_nii_path,
                                mask_foreground_value=1, convert_original_dicom=False)    

    ROIs = glob.glob(join(save_path, '**/**/mask_ROI*.nii.gz'), recursive=True)
    for ROI in ROIs:
        try:
            os.remove(ROI)
        except:
            print('do not find: ', ROI)

    print('finish: ', save_path)

    print('img: ', img_dcms_path)
    print('contours:', contour_path)

def convert_img(img_dcms_path, contour_path, save_path, img_type):
    os.makedirs(save_path, exist_ok=True)

    remove_DS_Store(img_dcms_path)
    remove_DS_Store(contour_path)

    os.makedirs(save_path, exist_ok=True)
    img_sitk = dcm2nii(img_dcms_path)
    if img_type == 'PET':
        sitk.WriteImage(img_sitk, join(save_path, 'PET_0001.nii.gz'))
        if args.annotations == 'PET':
            # save organ annotations according to PET
            convert_contours(contour_path, save_path, img_dcms_path)
    elif img_type == 'CT':
        sitk.WriteImage(img_sitk, join(save_path, 'CT_0000.nii.gz'))
        if args.annotations == 'CT':
            # save organ annotations according to CT
            convert_contours(contour_path, save_path, img_dcms_path)


os.makedirs(save_root_path, exist_ok=True)
remove_DS_Store(root_path)
hpis = sorted(os.listdir(root_path))
for hpi in hpis:
    hpi_path = join(root_path, hpi)
    remove_DS_Store(hpi_path)
    for inj in sorted(os.listdir(hpi_path)):
        contours_path = join(hpi_path, inj, 'ROI')
        remove_DS_Store(contours_path)
        ct_path = join(hpi_path, inj, 'CT')
        remove_DS_Store(ct_path)
        pet_path = join(hpi_path, inj, 'PET')
        remove_DS_Store(pet_path)
        convert_img(ct_path, contours_path, join(save_root_path, hpi, inj), 'CT')
        convert_img(pet_path, contours_path, join(save_root_path, hpi, inj), 'PET')
        