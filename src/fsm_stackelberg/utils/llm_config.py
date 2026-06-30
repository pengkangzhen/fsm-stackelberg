"""LangChain LLM configuration - adapted from mako/utils/user_proxy_config.py."""

import os
from typing import Optional, Union

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

load_dotenv()

# MAKO's LLM endpoints are all direct-connection public cloud APIs (see *_BASE_URL in
# .env); they must NOT be routed through a local HTTP proxy. Strip any inherited proxy
# environment variables so the httpx/openai/anthropic SDKs never attempt a local proxy
# (e.g. 127.0.0.1:7897) that may not be running — which otherwise causes
# ``ConnectError: [Errno 61] Connection refused`` on the very first LLM call.
for _proxy_var in ("http_proxy", "https_proxy", "all_proxy",
                   "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_proxy_var, None)

# Provider 到环境变量的映射（复用 .env 配置）
PROVIDER_ENV = {
    "DeepSeek": ("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"),
    "OpenAI": ("OPENAI_BASE_URL", "OPENAI_API_KEY"),
    "Anthropic": ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"),
    "Gemini": ("GEMINI_BASE_URL", "GEMINI_API_KEY"),
    "Qwen": ("QWEN_BASE_URL", "QWEN_API_KEY"),
    "OpenRouter": ("OPENROUTER_BASE_URL", "OPENROUTER_API_KEY"),
    "MiniMax": ("MINIMAX_ANTHROPIC_BASE_URL", "MINIMAX_API_KEY"),
    "ZhipuAI": ("ZHIPUAI_BASE_URL", "ZHIPUAI_API_KEY"),
    "DashScope": ("QWEN_BASE_URL", "QWEN_API_KEY"),  # 阿里云百炼，复用 Qwen 凭据
    "DashScopeGLM": ("DASHSCOPE_GLM_BASE_URL", "DASHSCOPE_GLM_API_KEY"),
    "Moonshot": ("MOONSHOT_BASE_URL", "MOONSHOT_API_KEY"),
    "MiMo": ("MIMO_BASE_URL", "MIMO_API_KEY"),
    "NVIDIA": ("NVIDIA_BASE_URL", "NVIDIA_API_KEY"),
}

PROVIDER_MODELS = {
    "DeepSeek": ["deepseek-v4-pro", "deepseek-v4-flash"],
    "OpenRouter": ["qwen/qwen3.5-plus-02-15","qwen/qwen3.6-plus-preview:free", "deepseek/deepseek-v3.2", "minimax/minimax-m2.5:free", "z-ai/glm-5", "moonshotai/kimi-k2.5", "moonshotai/kimi-k2.6:free", "openai/gpt-oss-120b:free"],
    "MiniMax": ["MiniMax-M2.7", "MiniMax-M3"],
    "ZhipuAI": ["glm-5", "glm-5.1"],
    "DashScope": ["glm-5.1", "qwen3.7-plus", "kimi-k2.6"],  # 阿里云百炼平台托管的模型
    "DashScopeGLM": ["glm-5.2"],
    "Moonshot": ["kimi-k2.5"],
    "MiMo": ["mimo-v2.5-pro"],
    "NVIDIA": ["z-ai/glm-5.1"],
    "Qwen": ["qwen3.7-plus"],
    "Gemini": ["gemini-3.1-pro"],
}

DEFAULT_PROVIDER = "MiMo"
DEFAULT_MODEL = "mimo-v2.5-pro"

# 使用 Anthropic SDK 的 provider（MiniMax M2.x 提供 Anthropic 兼容端点）
# M3 使用 OpenAI 兼容端点，不走 Anthropic 路径
ANTHROPIC_PROVIDERS = {"MiniMax"}
MINIMAX_OPENAI_MODELS = {"MiniMax-M3"}


def get_llm(
    provider: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
) -> Union[BaseChatModel, "ChatAnthropic"]:
    """Create LLM instance for the given provider.

    Args:
        provider: LLM provider name
        model: Model name (if None, uses first model in provider config)
        temperature: Temperature for generation

    Returns:
        ChatOpenAI or ChatAnthropic instance configured for the provider
    """
    if provider not in PROVIDER_ENV:
        available = list(PROVIDER_ENV.keys())
        raise ValueError(f"Unknown provider: {provider}. Available: {available}")

    base_url_env, api_key_env = PROVIDER_ENV[provider]
    base_url = os.getenv(base_url_env)
    api_key = os.getenv(api_key_env)

    if not base_url or not api_key:
        raise ValueError(
            f"Missing environment variables for {provider}: {base_url_env}, {api_key_env}"
        )

    if model is None:
        model = PROVIDER_MODELS[provider][0]

    # MiniMax M3 uses Anthropic endpoint with thinking enabled for tool calling
    if provider == "MiniMax" and model in MINIMAX_OPENAI_MODELS:
        from langchain_anthropic import ChatAnthropic

        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

        base = ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            anthropic_api_url=base_url,
            default_headers={"X-Api-Key": api_key},
            max_tokens=16384,
            thinking={"type": "enabled", "budget_tokens": 8192},
        )

        class PatchedMiniMaxM3(type(base)):
            def with_structured_output(self, schema, method=None, **kw):
                return super().with_structured_output(schema, method="function_calling", **kw)
        base.__class__ = PatchedMiniMaxM3

        return base

    # Anthropic-compatible providers (MiniMax M2.x)
    if provider in ANTHROPIC_PROVIDERS:
        from langchain_anthropic import ChatAnthropic

        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

        base = ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            anthropic_api_url=base_url,
            default_headers={"X-Api-Key": api_key},
            max_tokens=8192,
            thinking={"type": "disabled"},
        )

        # MiniMax M2.x doesn't support json_mode structured output.
        # Always use function_calling for MiniMax.
        if provider == "MiniMax":
            class PatchedMiniMaxChatAnthropic(type(base)):
                def with_structured_output(self, schema, method=None, **kw):
                    return super().with_structured_output(schema, method="function_calling", **kw)
            base.__class__ = PatchedMiniMaxChatAnthropic

        return base

    kwargs = dict(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        request_timeout=300,
        max_retries=2,
        max_tokens=max_tokens,
    )

    # GLM-5.1 on DashScope: thinking mode is on by default but incompatible
    # with function_calling (tool_choice=required). Disable it explicitly.
    # GLM-5.2 on DashScopeGLM: same issue — disable thinking.
    is_glm_model = "glm" in model.lower()
    if provider in ("DashScope", "DashScopeGLM") and is_glm_model:
        kwargs["extra_body"] = {"enable_thinking": False}

    # Use function_calling for providers that produce invalid JSON under json_mode
    _fc_providers = {"ZhipuAI", "MiMo", "DashScopeGLM"}
    # DashScope + Kimi: json_schema wraps output in markdown (> prefix), use function_calling
    is_kimi_model = "kimi" in model.lower()
    if provider == "DashScope" and is_kimi_model:
        _fc_providers = _fc_providers | {"DashScope"}
    if provider in _fc_providers:
        class PatchedFCChatOpenAI(ChatOpenAI):
            def with_structured_output(self, schema, method=None, **kw):
                return super().with_structured_output(schema, method="function_calling", **kw)
        return PatchedFCChatOpenAI(**kwargs)

    # DeepSeek v4+ thinking mode
    is_deepseek_v4 = provider == "DeepSeek" and "v4" in model.lower()
    if is_deepseek_v4:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    is_qwen3_model = "qwen3" in model.lower() or "qwen/qwen3" in model.lower()

    # Qwen3+ thinking mode: enable via DashScope (direct), disable via OpenRouter
    # (OpenRouter uses a different payload shape and may not support it reliably).
    # DashScope thinking mode needs longer timeout and streaming to avoid connection drops.
    # Qwen3+ thinking mode: MAKO's multi-agent architecture already decomposes
    # reasoning across agents, so internal thinking is redundant and causes
    # structured-output timeouts. Disable for all Qwen3 models.
    if is_qwen3_model:
        if provider == "Qwen":
            kwargs["extra_body"] = {"enable_thinking": False}
        elif provider == "DashScope":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:  # OpenRouter etc.
            kwargs["extra_body"] = {"thinking": {"enabled": False}}

    # Kimi K2.5 requires temperature=1
    if "kimi" in model.lower():
        kwargs["temperature"] = 1

    # OpenRouter + Kimi: streaming hangs, json_mode returns empty
    if provider == "OpenRouter" and "kimi" in model.lower():
        kwargs["streaming"] = False
        kwargs["request_timeout"] = 300

        class PatchedKimiChatOpenAI(ChatOpenAI):
            def with_structured_output(self, schema, method=None, **kw):
                return super().with_structured_output(schema, method="function_calling", **kw)
        return PatchedKimiChatOpenAI(**kwargs)

    # OpenRouter + Qwen: structured-output responses can stall or arrive as malformed
    # streamed payloads; use the non-streaming code path with a bounded timeout.
    if provider == "OpenRouter" and is_qwen3_model:
        kwargs["streaming"] = False
        kwargs["request_timeout"] = 300

    # OpenRouter + glm-5: json_schema returns empty intermittently; use function_calling
    if provider == "OpenRouter" and "glm" in model.lower():
        class PatchedGlmOpenRouterChatOpenAI(ChatOpenAI):
            def with_structured_output(self, schema, method=None, **kw):
                return super().with_structured_output(schema, method="function_calling", **kw)
        return PatchedGlmOpenRouterChatOpenAI(**kwargs)

    return ChatOpenAI(**kwargs)


def get_available_providers() -> list[str]:
    """Get list of available providers."""
    return list(PROVIDER_ENV.keys())


def get_provider_models(provider: str) -> list[str]:
    """Get list of available models for a provider."""
    if provider not in PROVIDER_MODELS:
        raise ValueError(f"Unknown provider: {provider}")
    return PROVIDER_MODELS[provider]
