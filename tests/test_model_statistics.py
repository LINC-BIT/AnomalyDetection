from fabric_defect_hub.model_statistics import parameter_counts


class _Parameter:
    def __init__(self, count, requires_grad):
        self.count = count
        self.requires_grad = requires_grad

    def numel(self):
        return self.count


class _Model:
    def parameters(self):
        return [_Parameter(3, True), _Parameter(5, False)]


def test_parameter_counts_includes_frozen_parameters():
    assert parameter_counts(_Model()) == {"parameter_count": 8, "trainable_parameter_count": 3}
