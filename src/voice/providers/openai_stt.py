"""OpenAI Whisper speech-to-text implementation."""

import logging
from typing import BinaryIO

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAISTT:
    """OpenAI Whisper STT provider."""

    def __init__(self, api_key: str | None = None):
        """Initialize OpenAI STT.

        Args:
            api_key: OpenAI API key (uses env var if not provided)

        Raises:
            Exception: If OpenAI client initialization fails
        """
        # 使用传入的 key 或环境变量创建 OpenAI 客户端
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        logger.info("OpenAI STT initialized")

    def transcribe(self, audio_file: BinaryIO) -> str:
        """Transcribe audio using OpenAI Whisper.

        Args:
            audio_file: Binary audio file

        Returns:
            Transcribed text (empty string on failure)

        Note:
            Errors are logged but not raised - returns empty string instead.
            This allows graceful degradation in user-facing applications.
        """
        try:
            # 将文件指针重置到开头（它可能已在别处被读取过）
            audio_file.seek(0)

            # 调用 OpenAI Whisper API 执行转录
            result = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )

            # 清理结果首尾空白
            transcribed = result.strip()
            logger.info(f"OpenAI STT: transcribed {len(transcribed)} chars")
            return transcribed

        except Exception as e:
            # 记录完整错误堆栈，便于调试
            logger.error(f"OpenAI STT failed: {e}", exc_info=True)
            # 返回空字符串，以便在面向用户的场景中优雅降级
            return ""
