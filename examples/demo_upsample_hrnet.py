import hydra
import torch

from src.networks.upsample_hrnet import HRNETConfig, HRNETUpsample


@hydra.main(
    version_base=None, config_path="../configs/network", config_name="upsample_hrnet"
)
def main(cfg: HRNETConfig):
    model = HRNETUpsample(
        stage1=cfg.stage1,
        stage2=cfg.stage2,
        stage3=cfg.stage3,
        stage4=cfg.stage4,
        bn_momentum=cfg.bn_momentum,
        dim=cfg.dim,
        upsample=cfg.upsample,
        upsample_scale_factor=cfg.upsample_scale_factor,
        interpolation_mode=cfg.interpolation_mode,
        pretrained=cfg.pretrained,
        pretrained_path=cfg.pretrained_path,
    )

    dummy_image_tensor = torch.randn(1, 3, 224, 224)

    desc_low_res, desc_upsampled = model(dummy_image_tensor)
    print(desc_low_res.shape)
    print(desc_upsampled.shape)


if __name__ == "__main__":
    main()
