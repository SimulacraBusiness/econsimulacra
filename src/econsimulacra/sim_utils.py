from __future__ import annotations

import json
import math
from random import Random
from typing import Optional, Type, Union

JsonValue = Union[dict, list, tuple, float, int]


def find_class(name: str, optional_class_list: Optional[list[Type]] = None) -> Type:
    _1 = __import__("econsimulacra", globals(), locals())
    _2 = __import__("econsimulacra.agents", globals(), locals(), ["*"])
    _3 = __import__("econsimulacra.envs", globals(), locals(), ["*"])
    _4 = __import__("econsimulacra.events", globals(), locals(), ["*"])
    _5 = __import__("econsimulacra.items", globals(), locals(), ["*"])
    _6 = __import__("econsimulacra.logs", globals(), locals(), ["*"])
    _7 = __import__("econsimulacra.llm_services", globals(), locals(), ["*"])
    _8 = __import__("econsimulacra.memory", globals(), locals(), ["*"])
    _9 = __import__("econsimulacra.social_networks", globals(), locals(), ["*"])
    _10 = __import__("econsimulacra.spaces", globals(), locals(), ["*"])
    candidates_spaces = [*globals().values(), *locals().values()]
    object_class_candidates = [
        getattr(m, name) for m in candidates_spaces if hasattr(m, name)
    ]
    if optional_class_list is not None:
        object_class_candidates.extend(
            [x for x in optional_class_list if x.__name__ == name]
        )
    if len(object_class_candidates) != 1:
        raise AttributeError(
            f"class for {name} is found {len(object_class_candidates)} times"
        )
    object_class: Type = object_class_candidates[0]
    return object_class


class JsonRandom:
    def __init__(self, prng: Random) -> None:
        self.prng: Random = prng

    def _next_uniform(self, min_value: float, max_value: float) -> float:
        return self.prng.random() * (max_value - min_value) + min_value

    def _next_normal(self, mu: float, sigma: float) -> float:
        return self.prng.gauss(mu=mu, sigma=sigma)

    def _next_exponential(self, lam: float) -> float:
        return lam * -math.log(self.prng.random())

    def random(self, json_value: JsonValue) -> float:
        if isinstance(json_value, (list, tuple)):
            if len(json_value) != 2:
                raise ValueError(
                    "Uniform distribution must be [min, max] but "
                    + json.dumps(json_value)
                )
            min_value: float = float(json_value[0])
            max_value: float = float(json_value[1])
            return self._next_uniform(min_value=min_value, max_value=max_value)
        if isinstance(json_value, dict):
            if len(json_value) != 1:
                raise ValueError(
                    "Multiple specification of distribution type: "
                    + json.dumps(json_value)
                )
            if "const" in json_value:
                args = json_value["const"]
                if not isinstance(args, list):
                    raise ValueError(
                        "Constant must be [value] (list) but " + json.dumps(json_value)
                    )
                if len(args) != 1:
                    raise ValueError(
                        "Constant must be [value] but " + json.dumps(json_value)
                    )
                value = float(args[0])
                return value
            if "uniform" in json_value:
                args = json_value["uniform"]
                if not isinstance(args, (list, tuple)):
                    raise ValueError(
                        "Uniform distribution must be [min, max] (list or tuple) but "
                        + json.dumps(json_value)
                    )
                if len(args) != 2:
                    raise ValueError(
                        "Uniform distribution must be [min, max] but "
                        + json.dumps(json_value)
                    )
                min_value = float(args[0])
                max_value = float(args[1])
                return self._next_uniform(min_value=min_value, max_value=max_value)
            if "normal" in json_value:
                args = json_value["normal"]
                if not isinstance(args, (list, tuple)):
                    raise ValueError(
                        "Normal distribution must be [mu, sigma] (list or tuple) but "
                        + json.dumps(json_value)
                    )
                if len(args) != 2:
                    raise ValueError(
                        "Normal distribution must be [mu, sigma] but "
                        + json.dumps(json_value)
                    )
                mu = float(args[0])
                sigma = float(args[1])
                return self._next_normal(mu=mu, sigma=sigma)
            if "expon" in json_value:
                args = json_value["expon"]
                if not isinstance(args, (list, tuple)):
                    raise ValueError(
                        "Exponential distribution must be [lambda] (list or tuple) but "
                        + json.dumps(json_value)
                    )
                if len(args) != 1:
                    raise ValueError(
                        "Exponential distribution must be [lambda] but "
                        + json.dumps(json_value)
                    )
                lam = float(args[0])
                return self._next_exponential(lam=lam)
            raise ValueError("Unknown distribution type: " + json.dumps(json_value))
        return float(json_value)
