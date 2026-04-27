"""Text-to-speech 工厂。

该模块提供一个工厂类，根据配置加载合适的 TTS provider。
"""

import logging
import os
from typing import Literal, cast

logger = logging.getLogger(__name__)

Provider = Literal["openai", "elevenlabs"]


class TextToSpeech:
    """Text-to-speech factory.

    Loads and delegates to specific TTS provider implementations.

    Example:
        >>> tts = TextToSpeech(provider="openai", voice="nova")
        >>> audio = tts.generate("Hello world")
        >>>
        >>> # Or from environment
        >>> tts = TextToSpeech.from_env()
        >>> if tts:
        ...     audio = tts.generate("Hello world")
    """

    def __init__(self, provider: Provider = "openai", api_key: str | None = None, **config):
        """Initialize TTS with specified provider.

        Args:
            provider: Provider name ("openai", "elevenlabs", etc.)
            api_key: API key (uses env var if not provided)
            **config: Provider-specific configuration
                OpenAI: voice="alloy", model="tts-1"
                ElevenLabs: voice_id="...", model_id="..."

        Raises:
            ValueError: If provider is unknown
        """
        self._provider_name = provider

        # 从参数或环境变量中解析 API key
        resolved_api_key = self._get_api_key(provider, api_key)

        # 加载并配置对应的 provider
        self._provider = self._load_provider(provider, resolved_api_key, config)

        logger.info(f"TextToSpeech created with provider={provider}")

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
            case "elevenlabs":
                return os.getenv("ELEVENLABS_API_KEY")
            case _:
                return None

    def _load_provider(self, provider: Provider, api_key: str | None, config: dict):
        """Load the appropriate TTS provider implementation.

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
                from voice.providers.openai_tts import OpenAITTS

                # 提取 OpenAI 专属配置，并设置默认值
                voice = config.get("voice", "alloy")
                model = config.get("model", "tts-1")

                return OpenAITTS(api_key=api_key, voice=voice, model=model)

            case "elevenlabs":
                # 未来扩展示例：如需添加 ElevenLabs 支持，实现 ElevenLabsTTS provider 后取消下面注释：
                # from voice.providers.elevenlabs_tts import ElevenLabsTTS
                # voice_id = config.get("voice_id")
                # model_id = config.get("model_id", "eleven_monolingual_v1")
                # return ElevenLabsTTS(api_key=api_key, voice_id=voice_id, model_id=model_id)
                raise NotImplementedError("ElevenLabs TTS provider not yet implemented")

            case _:
                # 兜底处理未知 provider
                raise ValueError(f"Unknown TTS provider: {provider}. Available providers: openai")

    @property
    def provider(self) -> str:
        """Get the provider name.

        Returns:
            Provider name string
        """
        return self._provider_name

    @classmethod
    def from_env(cls) -> "TextToSpeech | None":
        """Create TTS from environment variables.

        Reads VOICE_TTS_PROVIDER env var to determine which provider to use.
        Returns None if not configured.

        Returns:
            TextToSpeech instance or None

        Example:
            >>> # In .env: VOICE_TTS_PROVIDER=openai
            >>> tts = TextToSpeech.from_env()
            >>> if tts:
            ...     audio = tts.generate("Hello world")
        """
        provider = os.getenv("VOICE_TTS_PROVIDER")

        # 如果未设置 provider，则语音功能中的 TTS 视为关闭
        if not provider:
            logger.debug("VOICE_TTS_PROVIDER not set, TTS disabled")
            return None

        try:
            # 使用环境变量中的 provider 创建实例
            # 若 provider 非法，这里会触发 ValueError
            return cls(provider=cast(Provider, provider))
        except Exception as e:
            # 记录错误但不让应用崩溃，允许在无语音能力的情况下继续运行
            logger.error(f"Failed to create TTS provider: {e}", exc_info=True)
            return None

    def generate(self, text: str) -> bytes | None:
        """Generate speech from text.

        Delegates to the underlying provider implementation.

        Args:
            text: Text to convert to speech

        Returns:
            Audio bytes (format depends on provider), or None on failure
        """
        return self._provider.generate(text)

    def get_format(self) -> str:
        """Get audio format (MIME type) for this provider.

        Returns:
            MIME type string (e.g., "audio/mp3")
        """
        return self._provider.get_format()
