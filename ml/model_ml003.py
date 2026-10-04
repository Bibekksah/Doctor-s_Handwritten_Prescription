import torch.nn as nn
from torchvision import models


class TransferMedicineClassifier(nn.Module):

    def __init__(self, num_classes=78):

        super().__init__()

        # Pretrained ResNet-18
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Keep the pretrained feature extractor.
        # Replace the original 1000-class ImageNet classifier.
        input_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)