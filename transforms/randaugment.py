from optastra.transforms.randaugment import RandAugmentConfig, RandAugment
from optastra.transforms.ops import ALL_OP_NAMES     # all 14 ops (photometric + geometric); was `_OPS` before the 28 Sep refactor
from optastra.transforms import Transform


@Transform.register(config=RandAugmentConfig())
def randaugment_moderate(cfg):
    cfg.ops = list(ALL_OP_NAMES)
    cfg.magnitude = 5
    return RandAugment(cfg)


@Transform.register(config=RandAugmentConfig())
def randaugment_weak(cfg):
    cfg.ops = list(ALL_OP_NAMES)
    cfg.magnitude = 2
    return RandAugment(cfg)

