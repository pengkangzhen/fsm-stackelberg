"""Utils module for fsm_stackelberg."""

__all__ = ["get_llm", "get_available_providers", "get_provider_models"]


def __getattr__(name: str):
    if name in {"get_llm", "get_available_providers", "get_provider_models"}:
        from . import llm_config

        return getattr(llm_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
