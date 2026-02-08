def run(*args, **kwargs):
    from .train import run as _run

    return _run(*args, **kwargs)


__all__ = ["run"]
