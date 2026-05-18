import unittest

from cli_config import build_arg_parser


class BuildArgParserTests(unittest.TestCase):
    def test_defaults_preserve_second_order_unlearning(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.unlearn_method, "second_order")

    def test_accepts_new_gradient_based_unlearning_methods(self):
        parser = build_arg_parser()
        self.assertEqual(parser.parse_args(["--unlearn_method", "ga"]).unlearn_method, "ga")
        self.assertEqual(
            parser.parse_args(["--unlearn_method", "ga_guarded"]).unlearn_method,
            "ga_guarded",
        )


if __name__ == "__main__":
    unittest.main()
