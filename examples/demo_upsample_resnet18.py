import hydra
import torch

from networks.upsample_resnet import ResNetUpsample, ResNetUpsampleConfig


@hydra.main(
    version_base=None, config_path="../configs/network", config_name="upsample_resnet"
)
def main(cfg: ResNetUpsampleConfig):
    model = ResNetUpsample(
        backbone_type=cfg.backbone_type,
        pretrained=cfg.pretrained,
        remove_avg_pool_layer=cfg.remove_avg_pool_layer,
        full_conv=cfg.full_conv,
        upsample=cfg.upsample,
        upsample_dim=cfg.upsample_dim,
    )

    dummy_image_tensor = torch.randn(1, 3, 224, 224)
    out = model(dummy_image_tensor)
    print(out.shape)


if __name__ == "__main__":
    main()
