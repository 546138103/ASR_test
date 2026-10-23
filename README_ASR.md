# SenseVoiceSmall 终端语音转文字

## 启动

双击本目录 `start_asr.cmd`。等待“开始监听”后说话；终端显示 `[临时]`，停顿约 0.7 秒后输出 `[确认]`。按 `Ctrl+C` 停止采集，程序会处理剩余语音。双击脚本最后会等待按键，便于查看错误。

也可以在本目录 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe -B -u realtime_asr.py
```

当前环境使用 Python 3.14.0、FunASR 1.4.16、PyTorch 2.14.1；默认自动选用 GPU 或 CPU。本机当前 PyTorch 未检测到 CUDA，使用 CPU。不需要切换 Python，也不需要激活虚拟环境。

## 常用参数

列出设备，然后使用设备编号选择麦克风：

```powershell
.\.venv\Scripts\python.exe -B realtime_asr.py --list-devices
.\.venv\Scripts\python.exe -B -u realtime_asr.py --device 1 --language zh
```

响应偏慢时关闭临时结果，只在停顿后识别整句：

```powershell
.\.venv\Scripts\python.exe -B -u realtime_asr.py --partial-interval 0
```

声音小而识别不到时降低阈值；环境噪声导致不停识别时提高阈值：

```powershell
.\.venv\Scripts\python.exe -B -u realtime_asr.py --threshold 0.004
.\.venv\Scripts\python.exe -B -u realtime_asr.py --threshold 0.02 --silence 0.9
```

阈值基于归一化音频 RMS，并非固定分贝；默认 0.008，需按麦克风增益和噪声调整。当前使用简单能量端点检测，适合安静环境，未引入额外 FSMN-VAD 模型。连续讲话默认每 15 秒强制分段，切分处可能影响词句完整性。

临时结果默认每新增约 1.2 秒音频重新识别，可能被后续结果修正；音频积压时优先处理录音和最终结果，跳过临时刷新。实际延迟还包括推理耗时。SenseVoiceSmall 是离线模型，此程序实现的是分段重识别的伪流式，并非模型原生逐字流式。

## 模型和数据

首次运行从 ModelScope 下载 `iic/SenseVoiceSmall`，权重约 936 MB。模型保存在 `models/SenseVoiceSmall`，缓存保存在 `.cache`；随后直接使用本地模型，无需重复下载。使用 FunASR 内置模型实现，不执行模型仓库远程代码。

不保存麦克风录音和转写记录；推理在本地完成。安装和运行所用临时目录 `.cache/codex_temp_install`、`.cache/codex_temp_runtime` 留在本目录内，其中运行临时目录供以后使用，尚未执行清理。

识别现有音频（推荐 WAV，也支持本机 soundfile 可解码的 MP3）：

```powershell
.\.venv\Scripts\python.exe -B -u realtime_asr.py --file models\SenseVoiceSmall\example\zh.mp3
```

只下载并检查模型：

```powershell
.\.venv\Scripts\python.exe -B -u realtime_asr.py --download-only
```

## 故障处理

- 麦克风不能打开：检查 Windows“隐私和安全性 → 麦克风”是否允许桌面应用使用麦克风；通过 `--list-devices` 选择正确输入。
- 提示缓冲区溢出或丢帧：关闭其他占用麦克风的程序，使用 `--partial-interval 0` 重试。
- 模型下载失败：检查实际使用的代理或网络，重新启动可续传。下载未完成不会写入完成标志。
- 提示未安装 ffmpeg：麦克风输入和本机 soundfile 解码的样例不需要 ffmpeg，可以继续使用。
- 查看完整异常：PowerShell 设置 `$env:ASR_DEBUG='1'` 后重新运行。
- 补装依赖：使用本目录虚拟环境和 `requirements_asr.txt`；PyTorch 沿用原有构建，不强制升级或替换。

## 测试

```powershell
.\.venv\Scripts\python.exe -B -m unittest test_realtime_asr -v
.\.venv\Scripts\python.exe -B -u realtime_asr.py --duration 5 --partial-interval 0
```

短时采集用于验证设备链路；识别质量仍需要对着麦克风实际说话检查。

官方参考：
- https://github.com/QwenAudio/SenseVoice
- https://github.com/modelscope/FunASR/tree/main/examples/industrial_data_pretraining/sense_voice


## 本机验证结果（2026-10-23）

- 9 项测试通过，包括分段、尾句重试、录音溢出和纯标点噪声过滤；`pip check` 无依赖冲突。
- SenseVoiceSmall 权重约 936 MB 已下载完成，本地模型加载成功，权重全部匹配。
- 官方中文样例长 5.616 秒，本机 CPU 单次识别耗时约 0.53 秒；不含首次导入与模型加载，不代表所有语音的耗时。
- 默认 Realtek 麦克风 16 kHz 单声道短时采集通过，停止后正常关闭；尚需用户现场说话确认拾音效果、阈值及识别准确率。
