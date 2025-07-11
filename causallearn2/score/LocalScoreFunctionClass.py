from typing import Any, Callable, List


class LocalScoreClass(object):
    def __init__(
        self,
        data: Any,
        local_score_fun: Callable[[Any, int, List[int], Any], float],
        parameters=None,
    ):
        self.data = data
        self.local_score_fun = local_score_fun
        self.parameters = parameters
        self.score_cache = {}

    def score(self, i: int, PAi: List[int]) -> float:
        if i not in self.score_cache:
            self.score_cache[i] = {}

        hash_key = tuple(sorted(PAi))

        if not self.score_cache[i].__contains__(hash_key):
            self.score_cache[i][hash_key] = self.local_score_fun(self.data, i, PAi, self.parameters)

        return self.score_cache[i][hash_key]
