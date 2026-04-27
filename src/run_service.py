import asyncio
import logging
import sys

import uvicorn
from dotenv import load_dotenv

from core import settings

load_dotenv()

if __name__ == "__main__":
    root_logger = logging.getLogger()
    if root_logger.handlers:
        print(
            f"Warning: Root logger already has {len(root_logger.handlers)} handler(s) configured. "
            f"basicConfig() will be ignored. Current level: {logging.getLevelName(root_logger.level)}"
        )

    logging.basicConfig(level=settings.LOG_LEVEL.to_logging_level())
    # 在 Windows 系统上设置兼容的事件循环策略。
    # 在 Windows 上，默认的 ProactorEventLoop 可能会与某些异步数据库驱动
    # （例如 psycopg，即 PostgreSQL 驱动）产生兼容性问题。
    # WindowsSelectorEventLoopPolicy 兼容性更好，并可避免处理数据库连接时出现
    # “RuntimeError: Event loop is closed” 这类错误。
    # 该设置需要在启动应用服务器之前完成。
    # 详情可参考文档：
    # https://www.psycopg.org/psycopg3/docs/advanced/async.html#asynchronous-operations
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(
        "service:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_dev(),
        timeout_graceful_shutdown=settings.GRACEFUL_SHUTDOWN_TIMEOUT,
    )
