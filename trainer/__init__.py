import os
import glob
import importlib

_here = os.path.dirname(os.path.abspath(__file__))
_pyfiles = glob.glob(os.path.join(_here, "*.py"))
for f in _pyfiles:
    mod_name = os.path.basename(f).split(".")[0]
    if mod_name == "__init__":
        continue
    importlib.import_module("." + mod_name, __name__)

