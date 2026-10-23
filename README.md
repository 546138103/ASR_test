# ASR_test

基于 SenseVoiceSmall、FunASR 和 PyTorch 的本地语音转文字实验，支持终端麦克风实时分段转写。

## 运行入口

- `realtime_asr.py`：持续采集麦克风，显示临时结果并在停顿后确认整句。
- `start_asr.cmd`：Windows 双击启动入口，使用本目录 `.venv`。
- `realtime_asr_test.py`：先录制一句再识别的对照实验。
- `test_asr.py`：本地录音文件识别实验，需要自行提供 `test_yue.wav`。
- `test_realtime_asr.py`：主程序的自动化回归测试。

安装、参数和故障处理见 [README_ASR.md](README_ASR.md)。已验证环境：Python 3.14.0、FunASR 1.4.16、PyTorch 2.14.1。虚拟环境和下载模型不随仓库上传，新机器需要准备环境并下载模型。

```powershell
.\.venv\Scripts\python.exe -B -u realtime_asr.py
.\.venv\Scripts\python.exe -B -m unittest test_realtime_asr -v
```

SenseVoiceSmall 使用分段重识别实现伪流式，实际延迟取决于分段参数和硬件性能。

## 本地文件

`.venv/`、`.cache/`、`models/`、个人测试录音和 `iris_experiments/` 均由 `.gitignore` 排除。Iris 练习文件仅保留在本地文件夹；运行其训练脚本时应先进入该文件夹，以便相对路径输出仍保存在其中。

## ASR 接入语义服务器

用户选定的逐句版 realtime_asr_test.py 已接入 HTTP 语义服务。双击 start_asr_semantic.cmd；先在 Semantic Recognition 启动服务器。详见 [README_SEMANTIC.md](README_SEMANTIC.md)。
