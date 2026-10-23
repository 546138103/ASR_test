# ASR 与语义识别接入计划

目标：保留 realtime_asr_test.py 的录音和识别方式，将识别后的整句文字通过 HTTP JSON 发送到 Semantic Recognition 的 task3_willingness。

协议：POST /v1/tasks/willingness，child_answer 为必填字符串，system_prompt 为可选本轮提示；默认服务器使用 TASK_PROMPT。响应沿用 TaskResult：result、reason、evidence、system_prompt、model。失败使用 HTTP 错误，不能把网络或 API 错误当成 no/unclear。

架构：客户端 semantic_client.py 使用 Python 标准库，无 ARK_API_KEY；服务器 semantic_server.py 使用 FastAPI/Uvicorn 调用现有 evaluate_willingness。只传文字，不传录音。当前默认 127.0.0.1:8765；外部部署要求服务访问令牌并使用 HTTPS。令牌与方舟 API Key 是不同凭据。

文件：ASR 新增 semantic_client.py、test_semantic_client.py、start_asr_semantic.cmd；修改 realtime_asr_test.py 使识别返回文字、循环移入 main，支持 --text 联调和 --asr-only。语义识别新增 semantic_server.py、src/test_semantic_server.py、start_semantic_server.cmd，更新依赖与使用说明。

- [x] 先写客户端与服务器边界测试并确认缺少实现时失败。
- [x] 实现服务器和客户端，保留录音参数与已有注释。
- [x] 完成 HTTP 真连接测试、原有测试、无密钥错误与鉴权验证。
- [x] 更新运行文档并准备两个仓库的任务提交。

验证重点：Unicode 原文保留；错误不当成孩子判断；错误不泄漏上游异常；未授权请求不调用模型；禁止客户端从服务器重定向泄漏令牌。实际 ARK_API_KEY 当前未配置，真实模型调用需用户在服务器终端输入。


验证结果：客户端 13 项测试、服务器 13 项测试通过；两边 pip check 正常。使用 Uvicorn 和真实 HTTP 套接字连接现有客户端与服务器，空回答调用真实 evaluate_willingness 返回 unclear，错误令牌返回 401，未配置方舟 Key 的非空回答返回 503；测试服务已停止，没有遗留后台进程。

独立代码审查未发现当前 loopback 方案的阻断问题。复现并修复非法 URL 导致客户端退出的问题，新增回归测试通过。没有调用真实收费 API；用户需在服务器端输入 ARK_API_KEY 后完成非空文本判断验证。当前仍为原脚本的逐句同步流程，等待服务器期间不采集下一句。
