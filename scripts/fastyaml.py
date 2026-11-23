"""Speed patch: AISim's perf_database.get_database re-parses the system YAML on every call
(before its own cache check), ~90% of sweep time. Memoize yaml.load by (path, mtime) and
return a deep copy so semantics are unchanged. Import before running reproduce.py."""
import copy, os
import yaml as _yaml
from aisimulate_core.sdk import perf_database

class _YamlShim:
    def __init__(self):
        self._cache = {}
    def __getattr__(self, k):
        return getattr(_yaml, k)
    def load(self, stream, Loader=None):
        name = getattr(stream, 'name', None)
        if name is None or Loader is None:
            return _yaml.load(stream, Loader=Loader)
        key = (os.path.abspath(name), os.path.getmtime(name), Loader)
        if key not in self._cache:
            self._cache[key] = _yaml.load(stream, Loader=Loader)
        return copy.deepcopy(self._cache[key])

perf_database.yaml = _YamlShim()
