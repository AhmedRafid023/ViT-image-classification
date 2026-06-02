# Import all dataset modules to trigger registration with dassl's DATASET_REGISTRY
from . import imagenet          # noqa: F401
from . import oxford_pets       # noqa: F401
from . import oxford_flowers    # noqa: F401
from . import dtd               # noqa: F401
from . import food101           # noqa: F401
from . import eurosat           # noqa: F401
from . import caltech101        # noqa: F401
from . import fgvc_aircraft     # noqa: F401
from . import stanford_cars     # noqa: F401
from . import sun397            # noqa: F401
from . import ucf101            # noqa: F401
from . import imagenet_a        # noqa: F401
from . import imagenet_r        # noqa: F401
from . import imagenet_sketch   # noqa: F401
from . import imagenetv2        # noqa: F401

# Re-export dassl's DATASET_REGISTRY for convenience
from dassl.data.datasets import DATASET_REGISTRY  # noqa: F401
