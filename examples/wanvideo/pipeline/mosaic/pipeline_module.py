from .common import *
from .module_base import _MosaicModuleBase
from .module_forward import _ForwardMixin
from .module_materialize import _MaterializeMixin
from .module_mixins import (
    _ModelLoadingMixin,
    _PipelineInputMixin,
    _ScheduleMixin,
    _TrainQwenMixin,
)


class WanMosaicPipelineModule(
    _ForwardMixin,
    _MaterializeMixin,
    _PipelineInputMixin,
    _TrainQwenMixin,
    _ScheduleMixin,
    _ModelLoadingMixin,
    _MosaicModuleBase,
    DiffusionTrainingModule,
):
    pass
