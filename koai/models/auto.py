from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, model_type_to_module_name, getattribute_from_module
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES, CONFIG_MAPPING_NAMES
from collections import OrderedDict
import importlib


class LazyAutoMapping(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module("koai.models.biencoder")
        return getattribute_from_module(self._modules[module_name], attr)


MODEL_FOR_BI_ENCODER_MAPPING_NAMES = OrderedDict([
    ("bert", "BertForBiEncoder"),
])

MODEL_FOR_MULTI_ENCODER_MAPPING = LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_BI_ENCODER_MAPPING_NAMES
)


class AutoModelForBiEncoder(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTI_ENCODER_MAPPING