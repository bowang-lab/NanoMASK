from nnunet.training.loss_functions.poly_loss import Poly1FocalLoss
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_Loss_Poly1Focal(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)

        self.loss = Poly1FocalLoss(3, epsilon=1.0, alpha=0.5, gamma=2.0)