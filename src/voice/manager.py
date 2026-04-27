"""VoiceManager - Streamlit 集成层。

该模块提供语音功能在 Streamlit 中的专用 UI 集成。
所有 Streamlit 相关依赖都隔离在这里。
"""

import logging
from typing import Optional

import streamlit as st

from voice.stt import SpeechToText
from voice.tts import TextToSpeech

logger = logging.getLogger(__name__)


class VoiceManager:
    """语音功能的 Streamlit 便捷封装层。

    这个类为语音输入输出提供 Streamlit 专用方法。
    它负责处理 UI 反馈（如加载动画、报错提示），而实际的
    语音处理则委托给 STT 和 TTS 模块。

    Example:
        >>> voice = VoiceManager.from_env()
        >>>
        >>> if voice:
        ...     user_input = voice.get_chat_input()
        ...     if user_input:
        ...         with st.chat_message("ai"):
        ...             voice.render_message("Hello!")
    """

    def __init__(self, stt: SpeechToText | None = None, tts: TextToSpeech | None = None):
        """Initialize VoiceManager.

        Args:
            stt: SpeechToText instance (None to disable STT)
            tts: TextToSpeech instance (None to disable TTS)
        """
        self.stt = stt
        self.tts = tts

        logger.info(
            f"VoiceManager: STT={'enabled' if stt else 'disabled'}, "
            f"TTS={'enabled' if tts else 'disabled'}"
        )

    @classmethod
    def from_env(cls) -> Optional["VoiceManager"]:
        """Create VoiceManager from environment variables.

        Reads VOICE_STT_PROVIDER and VOICE_TTS_PROVIDER to configure
        speech-to-text and text-to-speech providers.

        Returns:
            VoiceManager if either STT or TTS is configured, None otherwise

        Example:
            >>> # In .env:
            >>> # VOICE_STT_PROVIDER=openai
            >>> # VOICE_TTS_PROVIDER=openai
            >>>
            >>> voice = VoiceManager.from_env()
            >>> # Returns configured VoiceManager or None if disabled
        """
        # 从环境变量中创建 STT 和 TTS
        stt = SpeechToText.from_env()
        tts = TextToSpeech.from_env()

        # 如果两者都未启用，则返回 None（表示没有语音功能）
        if not stt and not tts:
            logger.debug("Voice features not configured")
            return None

        return cls(stt=stt, tts=tts)

    def _transcribe_audio(self, audio) -> str | None:
        """Transcribe audio with UI feedback.

        Shows spinner during transcription and error message on failure.

        Args:
            audio: Audio file object from Streamlit chat input

        Returns:
            Transcribed text, or None if transcription failed
        """
        # 防御性检查（如果调用方式正确，理论上不应进入这里）
        if not self.stt:
            st.error("⚠️ Speech-to-text not configured.")
            return None

        # 转录过程中显示加载提示
        with st.spinner("🎤 Transcribing audio..."):
            transcribed = self.stt.transcribe(audio)

        # 检查转录是否成功
        if not transcribed:
            st.error("⚠️ Transcription failed. Please try again or type your message.")
            return None

        return transcribed

    def get_chat_input(self, placeholder: str = "Your message") -> str | None:
        """Get chat input with optional voice transcription.

        Handles Streamlit UI including audio input widget and transcription
        feedback (spinner, errors).

        Args:
            placeholder: Placeholder text for input

        Returns:
            User's message (transcribed if audio, otherwise text), or None if no input
        """
        # 没有 STT 时，使用普通文本输入
        if not self.stt:
            return st.chat_input(placeholder)

        # 启用 STT 时，使用支持音频的输入组件
        chat_value = st.chat_input(placeholder, accept_audio=True)

        if not chat_value:
            return None

        # 处理字符串返回值（纯文本输入）
        if isinstance(chat_value, str):
            return chat_value

        # 处理对象/字典返回值（支持音频输入）
        # 提取文本内容，同时兼容属性访问和字典访问
        text_content = None
        if hasattr(chat_value, "text"):
            text_content = chat_value.text
        elif isinstance(chat_value, dict):
            text_content = chat_value.get("text", "")

        # 提取音频内容，同时兼容属性访问和字典访问
        audio_content = None
        if hasattr(chat_value, "audio"):
            audio_content = chat_value.audio
        elif isinstance(chat_value, dict):
            audio_content = chat_value.get("audio")

        # 如果提供了音频，则先进行转录
        if audio_content:
            return self._transcribe_audio(audio_content)

        # 如果没有音频，则返回文本内容
        if text_content:
            return text_content

        # 既没有文本也没有音频
        return None

    def render_message(self, content: str, container=None, audio_only: bool = False) -> None:
        """Render message with optional TTS audio.

        Handles Streamlit UI including text display and audio player.
        Saves generated audio in session state so it persists across reruns.

        Args:
            content: Message content to display
            container: Streamlit container (defaults to current context)
            audio_only: If True, only render audio (text already displayed)
        """
        if container is None:
            container = st

        # 除非处于 audio_only 模式，否则先显示文本
        if not audio_only:
            container.write(content)

        # 如果启用了 TTS 且内容非空，则附加音频
        if self.tts and content.strip():
            # 生成音频时先显示占位提示
            placeholder = container.empty()
            with placeholder:
                st.caption("🎙️ Generating audio...")

            # 生成 TTS 音频
            audio = self.tts.generate(content)

            # 将音频保存到 session state 中，关联到最后一条 AI 消息
            # 这样在 st.rerun() 后也能继续保留
            if audio:
                st.session_state.last_audio = {"data": audio, "format": self.tts.get_format()}

            # 用音频播放器或错误提示替换占位内容
            if audio:
                placeholder.audio(audio, format=self.tts.get_format())
            else:
                placeholder.caption("🔇 Audio generation unavailable")
