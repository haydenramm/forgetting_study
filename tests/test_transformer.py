import unittest
import torch

from forgetting_study.model import Block, Transformer
from forgetting_study.config import ModelConfig

N_SAMPLES = 100  # number of dummy samples
EMBEDDING_D = 64  # dummy embedding dimension
VOCAB_SIZE = 100  # dummy vocab size

CFG = ModelConfig()


class TestBlock(unittest.TestCase):
    """Take a closer look at the transformer block class."""

    def setUp(self):
        self.transformer = Transformer(cfg=CFG, vocab_size=VOCAB_SIZE)
        self.block = Block(cfg=CFG, tap=self.transformer._tap)
        self.dummy_x = torch.randn(size=(N_SAMPLES, CFG.dim))
        return super().setUp()

    def test_input(self):
        raise NotImplementedError

    def test_output(self):
        raise NotImplementedError


class TestTwoBlockTransformer(unittest.TestCase):
    """Tests a transformer made of two blocks."""

    def setUp(self):
        self.transformer = Transformer()
        return super().setUp()


if __name__ == "__main__":
    unittest.main()
