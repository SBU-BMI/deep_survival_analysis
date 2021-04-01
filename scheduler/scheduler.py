import torch
import torch.utils.data
import torchvision.transforms as transforms

from utils import ensure_dir
import datasets.patch_data_uint8 as patch_data
import data_preprocess.data_preprocess as data_preprocess
import classifiers.patchcnn_em as patchcnn_em
import classifiers.wsi_cls as wsi_cls


# directory organization
# intermediate/
#     |--mask_root: string
#         |--patch_size_224_224_scale_1
#             |--fg_masks  # masks for round_no 0
#             |--disc_masks_round_1  # masks for round_no > 0
#             |--disc_masks_round_2  # masks for round_no > 0
#         |--patch_size_224_224_scale_2
#             |--fg_masks  # masks for round_no 0
#             |--disc_masks_round_1  # masks for round_no > 0
#             |--disc_masks_/round_2  # masks for round_no > 0
#     |--cnn_root: string
#         |--patch_size_224_224_scale_1
#             |--models
#         |--patch_size_224_224_scale_2
#             |--models
#     |--train_feature_root: string
#         |--patch_size_224_224_scale_1
#             |--{WSI_ID}_hist_feat.npy
#         |--patch_size_224_224_scale_2
#             |--{WSI_ID}_hist_feat.npy
#     |--test_feature_root: string
#         |--patch_size_224_224_scale_1
#             |--{WSI_ID}_hist_feat.npy
#         |--patch_size_224_224_scale_2
#             |--{WSI_ID}_hist_feat.npy
#     |--svm_root: string
#         |--linear
#             |--model.pkl
#         |--rbf
#             |--model.pkl
#


class Scheduler:
    def __init__(self, args, config, device):
        self.args = args
        self.device = device
        self.config = config
        self._load_arguments()
        self._prepare_intermediate_dirs()

    def _load_arguments(self):
        # data sources
        self.wsi_root = self.config['tile_process']['WSIs']['output_path']
        self.nu_seg_root = self.config['tile_process']['Nuclei_segs']['output_path']
        self.tumor_pred_root = self.config['tile_process']['Tumor_preds']['output_path']
        self.til_pred_root = self.config['tile_process']['TIL_preds']['output_path']
        self.label_file = self.config['tile_process']['label_file']
        
        # data loader parameters
        self.batch_size = self.config['data_loader']['batch_size']
        self.shuffle = self.config['data_loader']['shuffle']
        self.num_workers = self.config['data_loader']['num_workers']
        self.drop_last = self.config['data_loader']['drop_last']

        # training settings
        self.train_ratio = self.config['train_ratio']
        self.max_num_patches = self.config['max_num_patches']
        self.scales = self.config['scales'] # tuple
        self.use_rgb_only = self.config['use_rgb_only']
        self.intermediate = self.config['intermediate']
        self.ncores = self.args.ncores

        # patch cnn training parameters
        self.n_rounds = self.config['patchcnn_em']['n_rounds']
        self.patch_size = self.config['patchcnn_em']['patch_size']
        self.tile_size = self.config['tile_process']['tile_size']
        self.cnn_args = self.config['patchcnn_em']['m_step']
        self.cnn_args['seg_quantile'] = self.config['patchcnn_em']['e_step']['seg_quantile']
        self.cnn_args['smooth_sigma'] = self.config['patchcnn_em']['e_step']['smooth_sigma']
        self.cnn_args['seg_quantile'] = self.config['patchcnn_em']['e_step']['seg_quantile']

        # wsi classification parameters
        self.kernel = self.config['wsi_cls']['kernel']
        self.Cs = self.config['wsi_cls']['Cs']
        self.gammas = self.config['wsi_cls']['gammas']

        # transform list to tuple 
        self.tumor_pred_root = tuple(self.tumor_pred_root)
        self.til_pred_root = tuple(self.til_pred_root)
        self.patch_size = tuple(self.patch_size)
        self.tile_size = tuple(self.tile_size)
        self.scales = tuple(self.scales)
        self.Cs = tuple(self.Cs)
        self.gammas = tuple(self.gammas)

    def _prepare_intermediate_dirs(self):        
        self.mask_root = '{}/mask_root'.format(self.intermediate)
        self.cnn_root = '{}/cnn_root'.format(self.intermediate)
        self.train_feature_root = '{}/train_feature_root'.format(self.intermediate)
        self.test_feature_root = '{}/test_feature_root'.format(self.intermediate)
        self.svm_root = '{}/svm_root'.format(self.intermediate)
        ensure_dir(self.mask_root)
        ensure_dir(self.cnn_root)
        ensure_dir(self.train_feature_root)
        ensure_dir(self.test_feature_root)
        ensure_dir(self.svm_root)

    def _init_patch_datasets(self):
        transform_img = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        patch_dataset_train = patch_data.Patch_Data(
            wsi_root=self.wsi_root,
            nu_seg_root=self.nu_seg_root,
            tumor_pred_root=self.tumor_pred_root,
            til_pred_root=self.til_pred_root,
            label_file=self.label_file,
            transform=transform_img,
            mask_root=self.mask_root,
            scale=self.scales[0],
            round_no=0,
            is_train=True,
            train_ratio=self.train_ratio,
            rgb_only=self.use_rgb_only,
            patch_size=self.patch_size,
            tile_size=self.tile_size
        )

        # dataset for E step in EM
        patch_dataset_train_4E = patch_data.Patch_Data(
            wsi_root=self.wsi_root,
            nu_seg_root=self.nu_seg_root,
            tumor_pred_root=self.tumor_pred_root,
            til_pred_root=self.til_pred_root,
            label_file=self.label_file,
            transform=transform_img,
            mask_root=self.mask_root,
            scale=self.scales[0],
            round_no=0,
            is_train=True,
            train_ratio=self.train_ratio,
            rgb_only=self.use_rgb_only,
            patch_size=self.patch_size,
            tile_size=self.tile_size
        )

        patch_dataset_test = patch_data.Patch_Data(
            wsi_root=self.wsi_root,
            nu_seg_root=self.nu_seg_root,
            tumor_pred_root=self.tumor_pred_root,
            til_pred_root=self.til_pred_root,
            label_file=self.label_file,
            transform=transform_img,
            mask_root=self.mask_root,
            scale=self.scales[0],
            round_no=0,
            is_train=False,
            train_ratio=self.train_ratio,
            rgb_only=self.use_rgb_only,
            patch_size=self.patch_size,
            tile_size=self.tile_size
        )

        return patch_dataset_train, patch_dataset_train_4E, patch_dataset_test

    def _get_dir(self, root_name, scale):
        output_dir = '{}/patch_size_{}_{}_scale_{}'.format(root_name, self.patch_size[0], self.patch_size[1], scale)
        return output_dir    
        
    def train_mil(self):
        patch_dataset_train, patch_dataset_train_4E, patch_dataset_test = self._init_patch_datasets()

        wsi_ids_train = patch_dataset_train.get_wsi_id_no().keys()
        wsi_ids_test = patch_dataset_test.get_wsi_id_no().keys()

        no_wsi_id_train = patch_dataset_train.get_no_wsi_id()  # num to wsi_id (train)
        no_wsi_id_test = patch_dataset_test.get_no_wsi_id()  # num to wsi_id (test)
        no_wsi_id = {**no_wsi_id_train, **no_wsi_id_test}

        num_cls = patch_dataset_train.get_num_cls()
        
        fg_mask = data_preprocess.FG_Mask(self.wsi_root, self.nu_seg_root, self.tumor_pred_root, self.til_pred_root, self.mask_root, self.label_file, self.scales[0], self.patch_size, self.tile_size, self.max_num_patches)

        # fg_mask.filter_patches_parallel(self.ncores)
        
        random_seed = 42

        for scale in self.scales:
            
            ##########################################################
            ##        Stage 1: E-M training of CNN
            ##########################################################

            # fg_mask.set_scale(scale)
            # if self.args.fg:
            #     fg_mask.compute_fg_parallel(self.ncores)

            patch_dataset_train.set_scale(scale)
            patch_dataset_train_4E.set_scale(scale)
            patch_dataset_test.set_scale(scale)
            patch_dataset_train.set_round_no(0)
            patch_dataset_train_4E.set_round_no(0)
            patch_dataset_test.set_round_no(0)
            
            cnn_output_dir = self._get_dir(self.cnn_root, scale)
            mask_dir = self._get_dir(self.mask_root, scale)
            train_feat_dir = self._get_dir(self.train_feature_root, scale)
            test_feat_dir = self._get_dir(self.test_feature_root, scale)
            self.cnn_args['output_dir'] = cnn_output_dir
            self.cnn_args['mask_dir'] = mask_dir
            self.cnn_args['no_wsi_id'] = no_wsi_id
            self.cnn_args['num_cls'] = num_cls
            
            trainer = patchcnn_em.PatchCNN_EM(
                args=self.cnn_args,
                device=self.device
            )

            self.n_rounds = 1
            for round_i in range(self.n_rounds):

                train_loader = torch.utils.data.DataLoader(
                    patch_dataset_train,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self.num_workers,
                    drop_last=False
                )
                test_loader = torch.utils.data.DataLoader(
                    patch_dataset_test,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self.num_workers,
                    drop_last=False
                )
                # need to test to see which loader is faster
                # eval_loader = patch_dataset_train_4E.sequential_loader()

                ensure_dir('{}/disc_masks_round_{}'.format(mask_dir, round_i))
                trainer.set_round_no(round_i)
                trainer.set_train_loader(train_loader)
                trainer.set_test_loader(test_loader)
                trainer.m_step()  
                # trainer.e_step()
                return 
                patch_dataset_train.set_round_no(round_i)

            ##########################################################
            ##   Stage 2: Compute WSI feature
            ##########################################################
            eval_loader = torch.utils.data.DataLoader(
                patch_dataset_train_4E,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=self.drop_last
            )
            # need to test to see which loader is faster
            # eval_loader = patch_dataset_train_4E.sequential_loader()
            ensure_dir(train_feat_dir)
            trainer.hist_feat(eval_loader, train_feat_dir)

            test_loader = torch.utils.data.DataLoader(
                patch_dataset_test,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=self.drop_last
            )
            # need to test to see which loader is faster
            # eval_loader = patch_dataset_train_4E.sequential_loader()

            ensure_dir(test_feat_dir)
            trainer.hist_feat(test_loader, test_feat_dir)

        ##########################################################
        ##   Stage 2: Training SVM
        ##########################################################
        self.svm_cls = wsi_cls.WSI_cls(
            train_path=self.train_feature_root,
            test_path=self.test_feature_root,
            svm_root=self.svm_root,
            wsi_ids_train=wsi_ids_train,
            wsi_ids_test=wsi_ids_test,
            scales=self.scales
        )

        self.svm_cls.train(self.kernel, self.Cs, self.gammas)

    def test_mil(self):
        ap, acc = self.svm_cls.test()
        print('Testing Average Precision (AP): {0:.4f}, Accuracy: {1:.4f}'.format(ap, acc))

    

