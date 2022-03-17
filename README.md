# PyTorch UNet

My PyTorch implementation of the UNet architecture by Ronneberger et al. ( https://arxiv.org/abs/1505.04597 )

## Requirements

- Python3
- PyTorch

## Usage

Import like this:

```
import unet
model = UNet(in_channels=3, num_classes=10, init_weights=True)
```

and then use the model as you would any other PyTorch model. 
Softmax or Logits are not included, must use a loss function that includes them (e.g. torch.nn.BCEWithLogitsLoss) or apply them during training.
Input dimensions must be multiples of 16=2<sup>4</sup> for the upscaled feature maps to match up.
