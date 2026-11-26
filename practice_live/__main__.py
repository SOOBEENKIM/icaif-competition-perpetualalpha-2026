"""One refresh and inference pass. Does not start a service or place orders."""
from .refresh import refresh
from .infer import infer

if __name__ == "__main__":
    infer(refresh())
