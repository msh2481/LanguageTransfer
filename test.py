import unittest
from beartype import beartype as typed
import torch as t


class Slow(unittest.TestCase):
    @typed
    def setUp(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from language_modeling import seed_everything

        seed_everything(0)
        if not (hasattr(self, "model") and hasattr(self, "tokenizer")):
            model_name = "Mlxa/brackets-flat"
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @typed
    def test_sampling(self):
        from language_modeling import generate_sample

        result = generate_sample(self.model, self.tokenizer, "<2 <8 <10", 10)
        self.assertEqual(result, "8> 10> 2> <28 28> <169 169> <44 44> <20")
        result = generate_sample(self.model, self.tokenizer, "<2 <8 <10", 10)
        self.assertEqual(result, "10> 8> 2> <233 <158 233> 158> <118 118> <22")


class Fast(unittest.TestCase):
    @typed
    def test_spectrum(self):
        from language_modeling import spectrum, seed_everything
        seed_everything(0)
        n = 1000
        d = 10
        data = t.randn((n, 3)) @ t.randn((3, d)) + 0.1 * t.randn((n, 5)) @ t.randn((5, d))
        result = spectrum(data)
        self.assertEqual((result > 0.01).tolist(), [True] * 8 + [False] * 2)
        self.assertEqual(result.sum().item(), 2.7378175258636475)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    # suite.addTest(unittest.makeSuite(Slow))
    suite.addTest(unittest.makeSuite(Fast))
    unittest.TextTestRunner().run(suite)
