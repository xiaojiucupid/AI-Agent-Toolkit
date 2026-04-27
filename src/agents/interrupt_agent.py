import logging
from datetime import datetime
from typing import Any

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from core import get_model, settings

# 新增日志记录器
logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    birthdate: datetime | None


def wrap_model(
    model: BaseChatModel | Runnable[LanguageModelInput, Any], system_prompt: BaseMessage
) -> RunnableSerializable[AgentState, Any]:
    preprocessor = RunnableLambda(
        lambda state: [system_prompt] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


background_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant that tells users there zodiac sign.
Provide a one sentence summary of the origin of zodiac signs.
Don't tell the user what their sign is, you are just demonstrating your knowledge on the topic.
""")


async def background(state: AgentState, config: RunnableConfig) -> AgentState:
    """This node is to demonstrate doing work before the interrupt"""

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, background_prompt.format())
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


birthdate_extraction_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert at extracting birthdates from conversational text.

Rules for extraction:
- Look for user messages that mention birthdates
- Consider various date formats (MM/DD/YYYY, YYYY-MM-DD, Month Day, Year)
- Validate that the date is reasonable (not in the future)
- If no clear birthdate was provided by the user, return None
""")


class BirthdateExtraction(BaseModel):
    birthdate: str | None = Field(
        description="The extracted birthdate in YYYY-MM-DD format. If no birthdate is found, this should be None."
    )
    reasoning: str = Field(
        description="Explanation of how the birthdate was extracted or why no birthdate was found"
    )


async def determine_birthdate(
    state: AgentState, config: RunnableConfig, store: BaseStore
) -> AgentState:
    """This node examines the conversation history to determine user's birthdate, checking store first."""

    # 尝试获取 user_id，以便为每个用户单独存储数据
    user_id = config["configurable"].get("user_id")
    logger.info(f"[determine_birthdate] Extracted user_id: {user_id}")
    namespace = None
    key = "birthdate"
    birthdate = None  # 初始化 birthdate

    if user_id:
        # 在命名空间中使用 user_id，确保每个用户的数据唯一
        namespace = (user_id,)

        # 检查当前用户的出生日期是否已存在于 store 中
        try:
            result = await store.aget(namespace, key=key)
            # 兼容 store.aget 直接返回 Item 或返回列表两种情况
            user_data = None
            if result:  # 检查是否返回了任何内容
                if isinstance(result, list):
                    if result:  # 检查列表是否非空
                        user_data = result[0]
                else:  # 假定它直接返回的是 Item 对象
                    user_data = result

            if user_data and user_data.value.get("birthdate"):
                # 将 ISO 格式字符串还原为 datetime 对象
                birthdate_str = user_data.value["birthdate"]
                birthdate = datetime.fromisoformat(birthdate_str) if birthdate_str else None
                # 如果已经有出生日期，直接返回
                logger.info(
                    f"[determine_birthdate] Found birthdate in store for user {user_id}: {birthdate}"
                )
                return {
                    "birthdate": birthdate,
                    "messages": [],
                }
        except Exception as e:
            # 记录错误，或处理 store 暂时不可用的情况
            logger.error(f"Error reading from store for namespace {namespace}, key {key}: {e}")
            # 读取失败时继续执行提取流程
            pass
    else:
        # 如果没有 user_id，就无法可靠地存取用户级别的持久化数据。
        # 这里可以记录一下这种情况。
        logger.warning(
            "Warning: user_id not found in config. Skipping persistent birthdate storage/retrieval for this run."
        )

    # 如果没有从 store 中取到出生日期，则继续进行提取
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m.with_structured_output(BirthdateExtraction), birthdate_extraction_prompt.format()
    ).with_config(tags=["skip_stream"])
    response: BirthdateExtraction = await model_runnable.ainvoke(state, config)

    # 如果提取后仍未找到出生日期，则触发中断
    if response.birthdate is None:
        birthdate_input = interrupt(f"{response.reasoning}\nPlease tell me your birthdate?")
        # 使用新的输入重新执行提取流程
        state["messages"].append(HumanMessage(birthdate_input))
        # 注意：递归调用可能需要谨慎处理深度或状态更新问题
        return await determine_birthdate(state, config, store)

    # 找到出生日期后，将字符串转换为 datetime
    try:
        birthdate = datetime.fromisoformat(response.birthdate)
    except ValueError:
        # 如果解析失败，则要求用户进一步澄清
        birthdate_input = interrupt(
            "I couldn't understand the date format. Please provide your birthdate in YYYY-MM-DD format."
        )
        # 使用新的输入重新执行提取流程
        state["messages"].append(HumanMessage(birthdate_input))
        # 注意：递归调用可能需要谨慎处理深度或状态更新问题
        return await determine_birthdate(state, config, store)

    # 仅当存在 user_id 时，才存储新提取出的出生日期
    if user_id and namespace:
        # 将 datetime 转为 ISO 格式字符串，以便 JSON 序列化
        birthdate_str = birthdate.isoformat() if birthdate else None
        try:
            await store.aput(namespace, key, {"birthdate": birthdate_str})
        except Exception as e:
            # 记录错误，或处理 store 写入失败的情况
            logger.error(f"Error writing to store for namespace {namespace}, key {key}: {e}")

    # 返回确定好的出生日期（可能来自 store，也可能来自提取结果）
    logger.info(f"[determine_birthdate] Returning birthdate {birthdate} for user {user_id}")
    return {
        "birthdate": birthdate,
        "messages": [],
    }


response_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant.

Known information:
- The user's birthdate is {birthdate_str}

User's latest message: "{last_user_message}"

Based on the known information and the user's message, provide a helpful and relevant response.
If the user asked for their birthdate, confirm it.
If the user asked for their zodiac sign, calculate it and tell them.
Otherwise, respond conversationally based on their message.
""")


async def generate_response(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generates the final response based on the user's query and the available birthdate."""
    birthdate = state.get("birthdate")
    if state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
        last_user_message = state["messages"][-1].content
    else:
        last_user_message = ""

    if not birthdate:
        # 理想情况下，如果 determine_birthdate 正常工作并在必要时触发中断，就不应执行到这里。
        # 这里用于处理 birthdate 仍然缺失的情况。
        return {
            "messages": [
                AIMessage(
                    content="I couldn't determine your birthdate. Could you please provide it?"
                )
            ]
        }

    birthdate_str = birthdate.strftime("%B %d, %Y")  # 用于展示的格式

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m, response_prompt.format(birthdate_str=birthdate_str, last_user_message=last_user_message)
    )
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


# 定义图结构
agent = StateGraph(AgentState)
agent.add_node("background", background)
agent.add_node("determine_birthdate", determine_birthdate)
agent.add_node("generate_response", generate_response)

agent.set_entry_point("background")
agent.add_edge("background", "determine_birthdate")
agent.add_edge("determine_birthdate", "generate_response")
agent.add_edge("generate_response", END)

interrupt_agent = agent.compile()
interrupt_agent.name = "interrupt-agent"
