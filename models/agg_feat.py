import torch
from torch import nn
from torch import Tensor
import models.mobilenet as mobilenet


class AvgFeat(nn.Module):
    def __init__(self,
        input_nc: int = 3,
        n_intervals: int = 5,
        width_mult: float = 1.0
    ) -> None:
        super(AvgFeat, self).__init__()

        last_channel = 1280
        
        self.model = mobilenet.mobilenet_v2(pretrained=False, progress=True, input_nc=input_nc, num_classes=n_intervals)
        self.model_pred = mobilenet.mobilenet_v2(pretrained=False, progress=True, input_nc=input_nc, num_classes=n_intervals)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, n_intervals),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor, split: int = 3) -> Tensor:
        imgs, preds = x[:, :split, :, :], x[:, split:, :, :]
        imgs_feat_mean = self.model.get_mean_feature_diff(imgs)
        preds_feat_mean = self.model_pred.get_mean_feature_diff(preds)
        feat_mean = 0.5 * (imgs_feat_mean + preds_feat_mean)
        out = self.classifier(feat_mean)
        return out

    def aggregate_features_imgs(self, x: Tensor) -> None:
        return self.model.aggregate_features(x)

    def aggregate_features_preds(self, x: Tensor) -> None:
        return self.model_pred.aggregate_features(x)

    def aggregate_features(self, imgs: Tensor, preds: Tensor) -> None:
        self.model.aggregate_features(imgs)
        self.model_pred.aggregate_features(imgs)
        
    def get_output(self, activation: bool = True):
        imgs_feat_mean = self.model.get_mean_feature()
        preds_feat_mean = self.model_pred.get_mean_feature()
        feat_mean = 0.5 * (imgs_feat_mean + preds_feat_mean)
        x = self.classifier(feat_mean)
        if activation:
            x = self.softmax(x)
        return x

    def get_mean_feature_imgs(self) -> Tensor:
        return self.model.get_mean_feature()

    def get_mean_feature_preds(self) -> Tensor:
        return self.model_pred.get_mean_feature()

    def get_mean_feature(self) -> Tensor:
        return 0.5 * (self.model.get_mean_feature() + self.model_pred.get_mean_feature())

    def mean_feature_to_fc_imgs(self, activation: bool = True) -> Tensor:
        return self.model.mean_feature_to_fc(activation)

    def mean_feature_to_fc_preds(self, activation: bool = True) -> Tensor:
        return self.model_pred.mean_feature_to_fc(activation)

    def reset_features_imgs(self) -> None:
        return self.model.reset_features()

    def reset_features_preds(self) -> None:
        return self.model_pred.reset_features()

    def reset_features(self) -> None:
        self.model.reset_features()
        self.model_pred.reset_features()
        

class CatFeat(nn.Module):
    def __init__(self,
        input_nc: int = 3,
        n_intervals: int = 5,
        width_mult: float = 1.0
    ) -> None:
        super(CatFeat, self).__init__()

        last_channel = 2560

        self.model = mobilenet.mobilenet_v2(pretrained=False, progress=True, input_nc=input_nc, num_classes=n_intervals)
        self.model_pred = mobilenet.mobilenet_v2(pretrained=False, progress=True, input_nc=input_nc, num_classes=n_intervals)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, n_intervals),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor, split: int = 3) -> Tensor:
        imgs, preds = x[:, :split, :, :], x[:, split:, :, :]
        imgs_feat_mean = self.model.get_mean_feature_diff(imgs)
        preds_feat_mean = self.model_pred.get_mean_feature_diff(preds)
        feat_mean = torch.cat((imgs_feat_mean, preds_feat_mean), 1)
        out = self.classifier(feat_mean)
        return out

    def aggregate_features_imgs(self, x: Tensor) -> None:
        return self.model.aggregate_features(x)

    def aggregate_features_preds(self, x: Tensor) -> None:
        return self.model_pred.aggregate_features(x)

    def aggregate_features(self, imgs: Tensor, preds: Tensor) -> None:
        self.model.aggregate_features(imgs)
        self.model_pred.aggregate_features(imgs)

    def get_output(self, activation: bool = True):
        imgs_feat_mean = self.model.get_mean_feature()
        preds_feat_mean = self.model_pred.get_mean_feature()
        feat_mean = torch.cat((imgs_feat_mean, preds_feat_mean), 1) 
        x = self.classifier(feat_mean)
        if activation:
            x = self.softmax(x)
        return x
    
    def get_mean_feature_imgs(self) -> Tensor:
        return self.model.get_mean_feature()

    def get_mean_feature_preds(self) -> Tensor:
        return self.model_pred.get_mean_feature()

    def get_mean_feature(self) -> Tensor:
        return torch.cat((imgs_feat_mean, preds_feat_mean), 1) 

    def mean_feature_to_fc_imgs(self, activation: bool = True) -> Tensor:
        return self.model.mean_feature_to_fc(activation)

    def mean_feature_to_fc_preds(self, activation: bool = True) -> Tensor:
        return self.model_pred.mean_feature_to_fc(activation)

    def reset_features_imgs(self) -> None:
        return self.model.reset_features()

    def reset_features_preds(self) -> None:
        return self.model_pred.reset_features()

    def reset_features(self) -> None:
        self.model.reset_features()
        self.model_pred.reset_features()
    
