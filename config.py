import os
import json
from tensorforce import TensorForceError


class Configuration(object):
    def __init__(self, allow_defaults=True, **kwargs):
        self._config = kwargs
        self._accessed = {k: False for k, v in kwargs.items() if not isinstance(v, Configuration)}
        self.allow_defaults = allow_defaults

    def to_json(self, filename):
        with open(filename, "w") as fp:
            fp.write(json.dumps(self.as_dict()))

    def as_dict(self):
        d = dict()
        for k, v in self._config.items():
            if isinstance(v, Configuration):
                d[k] = v.as_dict()
            else:
                d[k] = v
        return d

    @staticmethod
    def from_json(filename, absolute_path=False, allow_defaults=True):
        if absolute_path:
            path = filename
        else:
            path = os.path.join(os.getcwd(), filename)
        with open(path, 'r', encoding='utf8') as fp:
            json_string = fp.read()
        return Configuration.from_json_string(json_string, allow_defaults=allow_defaults)

    @staticmethod
    def from_json_string(json_string, allow_defaults=True):
        config = json.loads(json_string)
        if "allow_defaults" in config and config["allow_defaults"] != allow_defaults:
            raise TensorForceError("allow_defaults conflict between JSON({}) and method call({}).".format(
                config["allow_defaults"],
                allow_defaults
            ))
        return Configuration(allow_defaults=allow_defaults, **config)

    def __str__(self):
        return '{' + ', '.join('{}.{}'.format(k, v) for k, v in self._config.items()) +'}'

    def __len__(self):
        return len(self._config)

    def __iter__(self):
        for key, value in self._config.items():
            if key in self._accessed:
                self._accessed[key] = True
            yield key, value

    def __contains__(self, key):
        return key in self._config

    def __getstate__(self):
        return self._config

    def __setstate__(self, d):
        self._config = d

    def __getattr__(self, key):
        if key not in self._config:
            raise TensorForceError("Value for {} is not defined.".format(key))
        if key in self._accessed:
            self._accessed[key] = True
        return self._config[key]

    def __setattr__(self, key, value):
        if key == '_config':
            value = {k: make_config_value(v) for k, v in value.items()}
            super(Configuration, self).__setattr__(key, value)
        elif key == '_accessed' or key == 'allow_defaults':
            super(Configuration, self).__setattr__(key, value)
        elif key not in self._config:
            raise TensorForceError("Value {} is not defined.".format(key))
        else:
            self._config[key] = make_config_value(value)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def items(self):
        return iter(self)

    def keys(self):
        return self._config.keys()

    def copy(self):
        return Configuration(**self._config)

    def default(self, defaults):
        for k, v in defaults.items():
            if k not in self._config:
                if not self.allow_defaults:
                    raise TensorForceError("This configuration not allow default.")
                if isinstance(v, dict):
                    v = Configuration(**v)
                else:
                    self._accessed[k] = False
                self._config[k] = v

    def not_accessed(self):
        not_accessed = list()
        for key, value in self._config.items():
            if isinstance(value, Configuration):
                for subkey in value.not_accessed():
                    not_accessed.append('{}.{}'.format(key, subkey))
            elif not self._accessed[key]:
                not_accessed.append(key)
        return not_accessed


def make_config_value(value):
    if isinstance(value, dict):
        return Configuration(**value)
    elif isinstance(value, list):
        return [make_config_value(v) for v in value]
    else:
        return value