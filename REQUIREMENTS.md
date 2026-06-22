# LLM API 轻量转发服务 — 需求文档

## 概述
一个轻量级的 OpenAI 兼容 API 反向代理服务，支持请求/响应 hook。

## 核心功能

### 1. 协议
- 代理 OpenAI 兼容格式：`/v1/chat/completions`、`/v1/models` 等
- 支持 streaming (SSE) 和非 streaming 响应

### 2. 多 Provider 配置
- 支持 JSON 配置多个 provider，每个 provider 包含：
  - `name`: provider 名称
  - `base_url`: 上游 API base URL
  - `api_key`: 上游 API key
  - `models`: 支持的模型列表
  - `extra_headers`: 额外请求头（静态的）
- 请求根据 model 名路由到对应 provider

### 3. 请求 Hook — x-auth-token 注入
- 某些 provider 需要额外的 `x-auth-token` 请求头
- token 从文件读取（如 `/var/run/llm-proxy/auth-token`）
- token 是动态的，由外部脚本定时刷新写入文件
- 服务启动时加载，支持热重载（文件变化时重新读取）
- 配置中可以指定 token 文件路径

### 4. 响应 Hook — reasoning_content 归一化
- 不同 provider 返回的 reasoning 字段名不统一
- 需要将响应中的 reasoning 内容统一归一化为 `reasoning_content` 字段
- 有些 provider 将 reasoning 内容直接放在 `content` 字段内（内联），需要提取出来
- 同时需要处理 streaming 场景下的归一化
- 非流式响应：检查 choices[].message 中的字段
- 流式响应：检查 SSE chunk 中的 delta 字段

### 5. 配置格式
- JSON 配置文件
- 示例结构：
```json
{
  "port": 8089,
  "providers": [
    {
      "name": "provider-a",
      "base_url": "https://api.provider-a.com",
      "api_key": "sk-xxx",
      "models": ["gpt-4o", "gpt-4o-mini"],
      "extra_headers": {},
      "auth_token_file": "/var/run/llm-proxy/provider-a-token"
    },
    {
      "name": "provider-b",
      "base_url": "https://api.provider-b.com",
      "api_key": "sk-yyy",
      "models": ["claude-sonnet-4"],
      "extra_headers": {"X-Custom-Header": "value"},
      "auth_token_file": null
    }
  ],
  "default_provider": "provider-a"
}
```

## 技术要求
- Python + FastAPI + httpx（异步 HTTP 客户端）
- 流式代理使用 SSE streaming
- 支持热重载配置文件
- systemd service 文件

## 不做的功能
- 负载均衡 / 多实例
- 用量统计 / 计费
- 调用方鉴权
- Mock / 缓存

## 测试要求
- 用子 agent 或 pytest 做基本测试
- 测试非流式请求转发
- 测试流式请求转发
- 测试 x-auth-token 注入
- 测试响应 reasoning_content 归一化（包括内联 reasoning 提取）
- 可以用 mock upstream 来测试
