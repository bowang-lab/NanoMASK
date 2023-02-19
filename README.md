# NanoMASK

A deep learning algorithm for auto-segmentation and pharmacokinetic quantitation for PET/CT functional imaging of nanomedicines


## Installation


1. Create a virtual environment

```bash
conda create -n nanomask python=3.8 -y
conda activate nanomask
```


2. Install required packages

```bash
# install pytorch following the official guideline
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
cd NanoMask 
pip install -e .
```

## Preprocessing your mouse PET/CT data
1. Convert PET/CT data format

    Run

    ```bash
    python dcm2nii.py -i input_path -o output_path -a CT
    ```

    - `input_path`: the input path contains three folders: 'CT' for CT data, 'PET' for PET data, and 'ROI' for organ contours.
    - `CT`: this can only be CT or PET, which represents converting organ contours according to CT data or PET data.

2. Register PET/CT data

    Install Greedy for image registration from https://greedy.readthedocs.io/en/latest/install.html.

    Use affine registration to register PET/CT data

    ```bash
    python affine_registration.py -i input_path -m CT
    ```

    - `input_path`: the input path contains PET, CT data and organ contours. All in '.nii.gz' format.
    - `CT`: this can only be CT or PET. 'CT' represents using CT data ('CT_0000.nii.gz') as the moving image and PET data ('PET_0001.nii.gz') as the fixed image. Apply Affine transformation to register CT and organ contours ('organs-CT.nii.gz') to PET. 'PET' represent using PET data as the moving image and CT data as the fixed image. 


## Organ segmentation for mouse PET/CT data

Download the [pretrained model](https://drive.google.com/drive/folders/17ymOMF_t4P7gwMBikWnsuq_fiLZ9xn2w?usp=share_link) and put it into `./NanoMask/nnunet/nnUNet_data/nnUNet/3d_fullres`. Run

```bash
nnUNet_predict -i input_path -o output_path -m 3d_fullres -t 006 -f 0
```

Notes:
- `input_path`: Please use absolute path and the path cannot have spaces. CT and PET data should have the suffix `_0000.nii.gz`, `_0001.nii.gz`, respectively. E.g., `Case1_0000.nii.gz, Case1_0001.nii.gz`
- `ouput_path`: the segmentation results will be saved in this path.


## Train the model based on your own data
Organize the dataset as follows

```
Task001_MousePETCT
    - imagesTr
        - case1_0000.nii.gz # CT
        - case1_0001.nii.gz # PET
    - imagesTs
    - labelsTr
        - case1.nii.gz # ground truth
    - dataset.json
```
Run

```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 Task001_MousePETCT 0 
```

## Evaluate your segmentation result
Run

```bash
python compute_metrics.py -s segmentation_path -r ground_truth_path -o output_path -n Heart Lungs Liver Spleen Kidneys Tumor
```

Notes:
- `Heart Lungs Liver Spleen Kidneys Tumor`: the organs in ascending order according to their mapped integers.
- `ouput_path`: the evaluation results, DSC and VD, will be saved in this path in a csv file.


## Evaluate your segmentation result
Run

```bash
python compute_metrics.py -s segmentation_path -r ground_truth_path -o output_path -n Heart Lungs Liver Spleen Kidneys Tumor
```

Notes:
- `n`: the mapped integers of each organ in ascending order
- `ouput_path`: the evaluation results, DSC and VD, will be saved in this path in a csv file.


## Acknowledgement
This customized model is based on the [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework. Thanks for the nnUNet team very much!

