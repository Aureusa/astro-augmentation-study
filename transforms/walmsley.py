from optastra.transforms import Transform, RandomHFlip, RandomVFlip, RandomRotation, RandomCrop, Compose
from optastra.transforms.geometric import RandomFlipConfig, RandomRotationConfig, RandomCropConfig
from dataclasses import dataclass


__all__ = ["WalmsleyTransforms", "ToGrayscale"]


@dataclass
class ToGrayscaleConfig:
    num_output_channels: int = 1


class ToGrayscale(Transform):
    """Collapse to grayscale by averaging across channels.
 
    Reproduces: "we average over the three colour channels to produce a
    single greyscale channel" (Walmsley et al. 2022, Sec 2.4) -- a plain
    mean, not a luma-weighted average (so deliberately different from
    torchvision's rgb_to_grayscale, which weights R/G/B by 0.2989/0.587/
    0.114). The paper does this to stop the model from using colour as a
    shortcut for morphology.
    """
 
    def __init__(self, cfg: ToGrayscaleConfig = ToGrayscaleConfig()):
        self.cfg = cfg
 
    def __call__(self, sample):
        image = sample.image
        gray = image if image.shape[0] == 1 else image.mean(dim=0, keepdim=True)
        if self.cfg.num_output_channels > 1:
            gray = gray.expand(self.cfg.num_output_channels, -1, -1)
        sample.image = gray
        return sample


@dataclass
class WalmsleyTransformsConfig:
    flip_p: float = 0.5
    rotation_degrees: tuple[float, float] = (0.0, 180.0)
    crop_size: int = 224
 
 
class WalmsleyTransforms(Transform):
    """Reproduces the augmentation recipe from:
 
        Walmsley et al. (2022), "Galaxy Zoo DECaLS: Detailed Visual
        Morphology Measurements from Volunteers and Deep Learning for
        314,000 Galaxies", MNRAS 509, 3966, Section 2.4.
 
    Recipe (applied in this order, resampled fresh on every load):
        1. random horizontal flip
        2. random vertical flip
        3. rotation by a random angle in (0, pi), missing pixels filled by
           reflection at the boundary
        4. crop about a random centroid to `crop_size` x `crop_size`
           (a slight, randomly off-centre zoom)
 
    The paper also applies this same pipeline at test time (as TTA, to
    marginalize over the augmentation distribution) -- that's a modeling/
    inference-loop decision rather than a transform, so it isn't handled
    here; wire it up wherever your eval loop builds its transform pipeline
    if you want to match the paper exactly.
    """
 
    def __init__(self, cfg: WalmsleyTransformsConfig = WalmsleyTransformsConfig()):
        self.cfg = cfg
        self._compose = Compose([
            RandomHFlip(RandomFlipConfig(p=cfg.flip_p)),
            RandomVFlip(RandomFlipConfig(p=cfg.flip_p)),
            RandomRotation(RandomRotationConfig(degrees=cfg.rotation_degrees)),
            RandomCrop(RandomCropConfig(size=cfg.crop_size)),
        ])
 
    def __call__(self, sample):
        return self._compose(sample)


@Transform.register(config=WalmsleyTransformsConfig())
def walmsley_transforms(cfg: WalmsleyTransformsConfig): return WalmsleyTransforms(cfg)

@Transform.register(config=ToGrayscaleConfig())
def to_grayscale(cfg: ToGrayscaleConfig): return ToGrayscale(cfg)
