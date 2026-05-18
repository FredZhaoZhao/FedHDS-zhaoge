from dataclasses import dataclass


@dataclass(frozen=True)
class UnlearningMethodSpec:
    name: str
    use_second_order: bool
    allow_direction_check: bool
    allow_guards: bool
    allow_retained_hessian: bool


UNLEARNING_METHOD_SPECS = {
    "second_order": UnlearningMethodSpec(
        name="second_order",
        use_second_order=True,
        allow_direction_check=True,
        allow_guards=True,
        allow_retained_hessian=True,
    ),
    "ga": UnlearningMethodSpec(
        name="ga",
        use_second_order=False,
        allow_direction_check=False,
        allow_guards=False,
        allow_retained_hessian=False,
    ),
    "ga_guarded": UnlearningMethodSpec(
        name="ga_guarded",
        use_second_order=False,
        allow_direction_check=False,
        allow_guards=True,
        allow_retained_hessian=False,
    ),
}


def get_unlearning_method_spec(name):
    try:
        return UNLEARNING_METHOD_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported unlearning method: {name}") from exc
