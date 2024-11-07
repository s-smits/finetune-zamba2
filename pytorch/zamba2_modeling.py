class Zamba2ForCausalLM(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


class Zamba2ForSequenceClassification(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


class Zamba2Model(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


class Zamba2PreTrainedModel(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


class ZoeDepthForDepthEstimation(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


class ZoeDepthPreTrainedModel(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


class Adafactor(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


class AdamW(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


def get_constant_schedule(*args, **kwargs):
    requires_backends(get_constant_schedule, ["torch"])


def get_constant_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_constant_schedule_with_warmup, ["torch"])


def get_cosine_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_cosine_schedule_with_warmup, ["torch"])


def get_cosine_with_hard_restarts_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_cosine_with_hard_restarts_schedule_with_warmup, ["torch"])


def get_inverse_sqrt_schedule(*args, **kwargs):
    requires_backends(get_inverse_sqrt_schedule, ["torch"])


def get_linear_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_linear_schedule_with_warmup, ["torch"])


def get_polynomial_decay_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_polynomial_decay_schedule_with_warmup, ["torch"])


def get_scheduler(*args, **kwargs):
    requires_backends(get_scheduler, ["torch"])


def get_wsd_schedule(*args, **kwargs):
    requires_backends(get_wsd_schedule, ["torch"])


class Conv1D(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


def apply_chunking_to_forward(*args, **kwargs):
    requires_backends(apply_chunking_to_forward, ["torch"])


def prune_layer(*args, **kwargs):
    requires_backends(prune_layer, ["torch"])


class Trainer(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])


def torch_distributed_zero_first(*args, **kwargs):
    requires_backends(torch_distributed_zero_first, ["torch"])


class Seq2SeqTrainer(metaclass=DummyObject):
    _backends = ["torch"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch"])
