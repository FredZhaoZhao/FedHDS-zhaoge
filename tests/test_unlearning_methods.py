import unittest

from unlearning_methods import get_unlearning_method_spec


class UnlearningMethodSpecTests(unittest.TestCase):
    def test_second_order_keeps_curvature_and_guards_available(self):
        spec = get_unlearning_method_spec("second_order")
        self.assertTrue(spec.use_second_order)
        self.assertTrue(spec.allow_direction_check)
        self.assertTrue(spec.allow_guards)
        self.assertTrue(spec.allow_retained_hessian)

    def test_ga_is_plain_gradient_ascent_without_guard_features(self):
        spec = get_unlearning_method_spec("ga")
        self.assertFalse(spec.use_second_order)
        self.assertFalse(spec.allow_direction_check)
        self.assertFalse(spec.allow_guards)
        self.assertFalse(spec.allow_retained_hessian)

    def test_ga_guarded_reuses_guards_without_retained_hessian(self):
        spec = get_unlearning_method_spec("ga_guarded")
        self.assertFalse(spec.use_second_order)
        self.assertFalse(spec.allow_direction_check)
        self.assertTrue(spec.allow_guards)
        self.assertFalse(spec.allow_retained_hessian)


if __name__ == "__main__":
    unittest.main()
