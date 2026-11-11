import os
from importlib import import_module


runtime = import_module(os.environ.get("PKGUARD_DM_RUNTIME_MODULE", "dm_runtime"))


def model_class():
    return getattr(runtime, os.environ.get("PKGUARD_DM_RUNTIME_CLASS", "DM"))
