```md
# AI Agent Service Toolkit 部署流程

## 一、前置准备
1. 克隆代码仓库至本地，进入项目根目录
```sh
git clone https://github.com/JoshuaC215/agent-service-toolkit.git
cd agent-service-toolkit
```

2. 配置环境变量
在项目根目录创建 `.env` 文件，至少需要配置一个大语言模型的 API 密钥。可参考项目内的 `.env.example` 文件，完整可配置项包含各类模型服务商 API 密钥、基于请求头的身份认证、LangSmith 链路追踪、测试与开发模式配置、OpenWeatherMap API 密钥等内容。
```sh
# 示例：配置 OpenAI API 密钥
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
```

## 二、Python 原生直接运行部署
推荐使用 uv 作为依赖管理工具，也可使用 pip 完成安装。
1. 安装 uv 包管理工具
```sh
curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
```

2. 安装项目依赖
```sh
# uv sync 会自动创建 .venv 虚拟环境
uv sync --frozen
```

3. 激活虚拟环境
```sh
source .venv/bin/activate
```

4. 启动 FastAPI 代理后端服务
```sh
python src/run_service.py
```

5. 新开终端，激活虚拟环境后启动 Streamlit 前端应用
```sh
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

6. 服务访问地址
- Streamlit 前端交互界面：默认地址 `http://localhost:8501`
- 代理服务 API 地址：`http://0.0.0.0:8080`
- API 在线文档地址：`http://0.0.0.0:8080/redoc`

## 三、Docker 方式部署
项目内置完整 Docker 配置，包含 postgres、agent_service、streamlit_app 三个核心服务，推荐使用 `docker compose watch` 实现开发阶段的自动热更新。

### 环境要求
需安装 Docker 及 Docker Compose（版本不低于 v2.23.0）

### 部署步骤
1. 从示例文件复制环境变量配置，补充必填的 LLM API 密钥
```sh
cp .env.example .env
# 编辑 .env 文件，完成相关密钥配置
```

2. 以监听模式构建并启动所有服务
```sh
docker compose watch
```
命令执行后会自动完成以下操作：
- 启动代理服务依赖的 PostgreSQL 数据库
- 启动基于 FastAPI 的 agent_service 代理核心服务
- 启动基于 Streamlit 的前端交互应用

3. 自动更新规则
- 项目内相关 Python 文件与目录的修改，会自动触发对应服务的更新
- 若修改了 `pyproject.toml` 或 `uv.lock` 依赖配置文件，需执行以下命令重新构建服务
```sh
docker compose up --build
```

4. 服务访问地址
- Streamlit 前端界面：浏览器访问 `http://localhost:8501`
- 代理服务 API 地址：`http://0.0.0.0:8080`
- API 在线文档地址：`http://0.0.0.0:8080/redoc`

5. 停止服务
```sh
docker compose down
```

## 四、无 Docker 本地开发环境部署
无需 Docker，直接通过 Python 虚拟环境搭建本地开发与运行环境。
1. 创建虚拟环境并安装项目依赖
```sh
uv sync --frozen
source .venv/bin/activate
```

2. 启动 FastAPI 后端服务
```sh
python src/run_service.py
```

3. 新开终端，激活虚拟环境后启动 Streamlit 前端应用
```sh
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

4. 访问应用
打开浏览器，访问 Streamlit 输出的地址，默认地址为 `http://localhost:8501`。

## 五、LangGraph Studio 开发环境部署
项目支持 LangGraph Studio（LangGraph 官方代理开发 IDE），可通过以下步骤完成配置与启动。
1. 完成前置准备中的环境变量配置，在项目根目录补充 `.env` 文件。
2. 执行 `uv sync` 时会自动安装 `langgraph-cli[inmem]` 依赖，无需额外安装。
3. 执行以下命令启动 LangGraph Studio
```sh
langgraph dev
```
4. 可根据开发需求，自定义项目根目录下 `langgraph.json` 配置文件。

## 六、自定义代理的部署配置
如需开发并部署自定义代理，按以下步骤操作：
1. 在 `src/agents` 目录中新增自定义代理文件，可参考目录内 `research_assistant.py` 或 `chatbot.py` 进行修改，自定义代理的行为与工具能力。
2. 在 `src/agents/agents.py` 文件中，导入并将新增的代理添加到 `agents` 字典中，配置完成后可通过 `/<your_agent_name>/invoke` 或 `/<your_agent_name>/stream` 路径调用该代理。
3. 根据自定义代理的能力，调整 `src/streamlit_app.py` 中的 Streamlit 前端交互界面，匹配代理的功能特性。

## 七、私有凭证文件处理
若你的代理或选用的大语言模型需要基于文件的凭证文件或证书，可使用项目内的 `privatecredentials/` 目录。该目录内除 `.gitkeep` 文件外的所有内容，均会被 git 与 docker 构建流程忽略，不会被提交至代码仓库或打包至镜像中。
```