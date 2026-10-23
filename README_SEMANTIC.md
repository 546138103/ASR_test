# ASR 客户端接入参与意愿判断

## 本机使用：先开服务器，再开客户端

1. 在 `Semantic Recognition` 项目双击 `start_semantic_server.cmd`。如果服务器进程没有 `ARK_API_KEY`，终端会要求输入方舟 API Key，输入不回显，不保存到文件。看到 Uvicorn 启动成功后保留该窗口。
2. 回到本项目，双击 `start_asr_semantic.cmd`，或者运行：

```powershell
.\.venv\Scripts\python.exe -B -u realtime_asr_test.py
```

本轮问题默认是“你想和他们一起玩吗？”。对着麦克风说“想”“不想”或“不知道”等回答；程序先输出转写文字，再输出服务器返回的 `result` 和 `reason`。

本次保留用户选择的录音实现及阈值：音量阈值 0.05、句末静音 0.8 秒、单句最多 10 秒。说话小声而未触发时，可自行调整 `VOLUME_THRESHOLD`。语义请求和 ASR 同步执行，等待判断期间不录音；请等下次显示“正在监听麦克风”再说下一句。

原来的 `start_asr.cmd` 仍然启动 `realtime_asr.py`，没有改成此联调入口。

## 不用麦克风，先测试文字

服务器运行后：

```powershell
.\.venv\Scripts\python.exe -B -u realtime_asr_test.py --text "我想和他们一起玩"
```

这条命令不加载 ASR 模型、不打开麦克风，适合先确认 API Key 和服务器可用。

只测试本脚本原有 ASR：

```powershell
.\.venv\Scripts\python.exe -B -u realtime_asr_test.py --asr-only
```

指定本轮人工提示：

```powershell
.\.venv\Scripts\python.exe -B -u realtime_asr_test.py --prompt "小朋友，你想和他们一起搭积木吗？"
```

## 传输内容

流程：客户端麦克风 → 本地 SenseVoiceSmall → 整句文字 → HTTP JSON → 服务器 `evaluate_willingness` → 结构化结果 → 客户端显示。

请求为 `POST http://127.0.0.1:8765/v1/tasks/willingness`：

```json
{"child_answer":"我想一起玩"}
```

可选 `system_prompt`；省略时由服务器提供 `task3_willingness.TASK_PROMPT`。成功结果包含 `result`、`reason`、`evidence`、`system_prompt`、`model`。`yes` 只表示愿意参加，不表示已经请求加入或完成游戏任务。

只发送文字，不发送音频或方舟 API Key。网络失败、服务器未配密钥、上游 API 失败都显示“未获得判断”，不把服务异常当成孩子的 `no` 或 `unclear`。请求不自动重试，避免重复调用收费 API。默认客户端网络超时 120 秒；上游服务仍沿用原任务代码的超时及重试设置。

## 后续切换真实服务器

服务器端设置 `ARK_API_KEY`，并设置独立的 `SEMANTIC_SERVICE_TOKEN` 后启动：

```powershell
$env:SEMANTIC_SERVICE_TOKEN = '<service-access-token>'
.\.venv\Scripts\python.exe -B -u semantic_server.py --host 0.0.0.0 --port 8765
```

随后为服务器配置 HTTPS 入口；客户端只配置服务器地址与服务访问令牌：

```powershell
$env:SEMANTIC_SERVER_URL = 'https://your-server.example'
$env:SEMANTIC_SERVICE_TOKEN = '<service-access-token>'
.\.venv\Scripts\python.exe -B -u realtime_asr_test.py
```

服务令牌与方舟 API Key 不是同一个密钥。客户端不需要方舟 SDK 或方舟 Key。远程部署的域名、证书、网络开放与正式用户鉴权尚未配置；当前完成的是本机客户端/服务器分离联调。客户端会拒绝非本机的明文 HTTP，并拒绝 HTTP 重定向，避免令牌转发到其他地址。

## 验证

```powershell
.\.venv\Scripts\python.exe -B -m unittest test_semantic_client test_realtime_asr -v
```

已验证 HTTP 原文传输、服务鉴权、未配置密钥时的错误，以及既有 ASR 自动测试。真实非空文本的豆包判断需要在服务器配置 `ARK_API_KEY` 后使用 `--text` 命令验证。
