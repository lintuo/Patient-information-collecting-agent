# Patient Agent

极简线上挂号前预问诊 Agent。

当前版本不使用前端、不使用 LangGraph、不使用数据库。核心流程是：

```text
患者输入 -> 主分诊 Agent -> 科室 Skill 子 Agent -> 可回退重分诊 -> 报告生成 -> 写入患者记忆 -> 可选一键挂号
```

## 配置

填写根目录 `.env`：

```bash
OPENAI_API_KEY=你的 API key
OPENAI_MODEL=deepseek-ai/DeepSeek-V3.2
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_API_MODE=chat_completions
PATIENT_AGENT_MAX_SPECIALTY_TURNS=4
```

如果使用官方 OpenAI，可以把 `OPENAI_BASE_URL` 留空，并把 `OPENAI_API_MODE` 设置为 `auto` 或 `responses`。代码也兼容 `OPENAI_MODEL_URL` 作为 `OPENAI_BASE_URL` 的别名。

## 运行

```bash
python -m patient_agent.simple_agent.app --patient-id patient_001
```

## 输出

流程结束时，医生预问诊报告只会保存到 `data/reports/{case_id}.md`，不会直接打印在命令行里。随后系统会询问是否需要一键挂号，输入 `y`、`yes` 或 `是` 会模拟挂号成功。

## 说明

项目逻辑见：

```text
docs/simple_agent_logic.md
```

测试 case 见：

```text
tests/cases/README.md
```



