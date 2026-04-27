"""Speech-to-text 工厂。

该模块提供一个工厂类，根据配置加载合适的 STT provider。
"""

import logging
import os
from typing import BinaryIO, Literal, cast

logger = logging.getLogger(__name__)

Provider = Literal["openai", "deepgram"]


class SpeechToText:
    """Speech-to-text factory.

    Loads and delegates to specific STT provider implementations.

    Example:
        >>> stt = SpeechToText(provider="openai")
        >>> text = stt.transcribe(audio_file)
        >>>
        >>> # Or from environment
        >>> stt = SpeechToText.from_env()
        >>> if stt:
        ...     text = stt.transcribe(audio_file)
    """

    def __init__(self, provider: Provider = "openai", api_key: str | None = None, **config):
        """Initialize STT with specified provider.

        Args:
            provider: Provider name ("openai", "deepgram", etc.)
            api_key: API key (uses env var if not provided)
            **config: Provider-specific configuration

        Raises:
            ValueError: If provider is unknown
        """
        self._provider_name = provider

        # 从参数或环境变量中解析 API key
        resolved_api_key = self._get_api_key(provider, api_key)

        # 加载并配置对应的 provider
        self._provider = self._load_provider(provider, resolved_api_key, config)

        logger.info(f"SpeechToText created with provider={provider}")

    def _get_api_key(self, provider: Provider, api_key: str | None) -> str | None:
        """Get API key from parameter or environment.

        Args:
            provider: Provider name
            api_key: API key from parameter (takes precedence)

        Returns:
            Resolved API key or None
        """
        # 如果显式传入了 API key，就优先使用它
        if api_key:
            return api_key

        # 否则根据 provider 从环境变量中读取
        match provider:
            case "openai":
                return os.getenv("OPENAI_API_KEY")
            case "deepgram":
                return os.getenv("DEEPGRAM_API_KEY")
            case _:
                return None

    def _load_provider(self, provider: Provider, api_key: str | None, config: dict):
        """Load the appropriate STT provider implementation.

        Args:
            provider: Provider name
            api_key: Resolved API key
            config: Provider-specific configuration

        Returns:
            Provider instance

        Raises:
            ValueError: If provider is unknown
            NotImplementedError: If provider not yet implemented
        """
        match provider:
            case "openai":
                from voice.providers.openai_stt import OpenAISTT

                return OpenAISTT(api_key=api_key, **config)

            case "deepgram":
                # 未来扩展示例：如需添加 Deepgram 支持，实现 DeepgramSTT provider 后取消下面注释：
                # from voice.providers.deepgram_stt import DeepgramSTT
                # return DeepgramSTT(api_key=api_key, **config)
                raise NotImplementedError("Deepgram STT provider not yet implemented")

            case _:
                # 兜底处理未知 provider
                raise ValueError(f"Unknown STT provider: {provider}. Available providers: openai")

    @property
    def provider(self) -> str:
        """Get the provider name.

        Returns:
            Provider name string
        """
        return self._provider_name

    @classmethod
    def from_env(cls) -> "SpeechToText | None":
        """Create STT from environment variables.

        Reads VOICE_STT_PROVIDER env var to determine which provider to use.
        Returns None if not configured.

        Returns:
            SpeechToText instance or None

        Example:
            >>> # In .env: VOICE_STT_PROVIDER=openai
            >>> stt = SpeechToText.from_env()
            >>> if stt:
            ...     text = stt.transcribe(audio_file)
        """
        provider = os.getenv("VOICE_STT_PROVIDER")

        # 如果未设置 provider，则语音功能中的 STT 视为关闭
        if not provider:
            logger.debug("VOICE_STT_PROVIDER not set, STT disabled")
            return None

        try:
            # 使用环境变量中的 provider 创建实例
            # 若 provider 非法，这里会触发 ValueError
            return cls(provider=cast(Provider, provider))
        except Exception as e:
            # 记录错误但不让应用崩溃，允许在无语音能力的情况下继续运行
            logger.error(f"Failed to create STT provider: {e}", exc_info=True)
            return None

    def transcribe(self, audio_file: BinaryIO) -> str:
        """Transcribe audio to text.

        Delegates to the underlying provider implementation.

        Args:
            audio_file: Binary audio file

        Returns:
            Transcribed text (empty string on failure)
        """
        return self._provider.transcribe(audio_file)
