"""lmms-eval wrapper for Cambrian in the interventions (scripts/interventions/multimodal.py).

Same prompts and scoring as the benchmark wrapper (cambrian.py), with eager attention (the ablation
patches of patch_cambrian.py edit the 4-D attention mask), no KV cache during generation (every
step re-runs the full sequence through the patched forward) and the device taken from the device map.
"""

from typing import Optional, Union

from utils.models.cambrian.cambrian import CambrianCustom as _CambrianBenchmarkWrapper
from utils.models.cambrian.cambrian_utils import get_cambrian_device


class CambrianCustom(_CambrianBenchmarkWrapper):
    """Cambrian for lmms-eval with the settings of the interventions."""

    generate_use_cache = False

    def __init__(
        self,
        pretrained: str = "nyu-visionx/cambrian-8b",
        device: Optional[str] = "cuda",
        device_map: str = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        cache_dir: Optional[str] = None,
        attn_implementation: str = "eager",
        **kwargs,
    ) -> None:
        self.attn_implementation = attn_implementation
        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )
        self._device = get_cambrian_device(self._model)
