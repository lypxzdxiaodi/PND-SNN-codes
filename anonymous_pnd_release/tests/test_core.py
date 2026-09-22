import unittest

import torch

from pnd.model import LAYER_TYPES, ModelConfig, PNDSequenceClassifier
from pnd.raw_snn_s4 import (
    Snn_s4,
    Snn_s4_linear,
    bidir_psnns4,
    bsnns4,
    bsnns4_linear,
    psnns4,
)


class CoreTest(unittest.TestCase):
    def test_all_compatibility_layers(self):
        x = torch.randn(2, 8, 32)
        for layer_type in (
            Snn_s4,
            bsnns4,
            psnns4,
            bidir_psnns4,
            Snn_s4_linear,
            bsnns4_linear,
        ):
            layer = layer_type(8, d_state=8, dropout=0.0)
            output, state = layer(x)
            self.assertEqual(output.shape, x.shape)
            self.assertIsNone(state)
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertIsNotNone(layer.kernel.C.grad)

    def test_all_public_variants(self):
        for name in LAYER_TYPES:
            config = ModelConfig(
                input_dim=1,
                num_classes=3,
                d_model=8,
                d_state=8,
                n_layers=2,
                layer=name,
            )
            model = PNDSequenceClassifier(config)
            output = model(torch.randn(2, 32, 1))
            self.assertEqual(output.shape, (2, 3))
            self.assertTrue(torch.isfinite(output).all())

    def test_hard_forward_is_binary_before_projection(self):
        layer = psnns4(8, d_state=8)
        values = []
        handle = layer.dropout.register_forward_pre_hook(
            lambda _module, inputs: values.append(inputs[0].detach())
        )
        try:
            layer(torch.randn(2, 8, 32))
        finally:
            handle.remove()
        self.assertTrue(torch.all((values[0] == 0) | (values[0] == 1)))


if __name__ == "__main__":
    unittest.main()
