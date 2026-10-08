from importlib.metadata import PackageNotFoundError, version

try:
    VERSION = version("aikraft")
except PackageNotFoundError:  # running from a source tree without an install
    VERSION = "0.0.0"
