import hydra
import torch

from src.backbones.resnet.resnet import ResNet


@hydra.main(version_base=None, config_path="../configs/resnet", config_name="resnet18")
def main(cfg):
    model = ResNet(
        block_type=cfg.block_type,
        layers=cfg.layers,
        num_classes=cfg.num_classes,
        remove_avg_pool_layer=cfg.remove_avg_pool_layer,
        full_conv=cfg.full_conv,
    )
    dummy_image_tensor = torch.randn(1, 3, 224, 224)
    out = model(dummy_image_tensor)
    print(out.shape)


if __name__ == "__main__":
    main()
