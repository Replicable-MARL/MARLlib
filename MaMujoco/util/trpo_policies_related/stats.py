# pylint:disable=missing-module-docstring
import functools
from typing import Any, Callable

def learner_stats(func: Callable[[Any], dict]) -> Callable[[Any], dict]:
    """Wrap function to return stats under learner stats key."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        stats = func(*args, **kwargs)
        nested = stats.get(LEARNER_STATS_KEY, {})
        unnested = {k: v for k, v in stats.items() if k != LEARNER_STATS_KEY}
        return {LEARNER_STATS_KEY: {**nested, **unnested}}

    return wrapped
