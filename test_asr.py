
from pathlib import Path
import time

from modelscope import snapshot_download
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


# ===== 1. 确定项目路径 =====
project_dir = Path(__file__).resolve().parent

# 模型准备保存的位置
save_dir = project_dir / "models" / "SenseVoiceSmall"

# 录音文件位置
audio_path = project_dir / "test_yue.wav"


# ===== 2. 下载模型 =====
print("1. 正在检查或下载 SenseVoiceSmall...")

model_dir = snapshot_download(
    "iic/SenseVoiceSmall",
    local_dir=str(save_dir)
)

print("预定保存位置：", save_dir)
print("实际模型目录：", model_dir)


# ===== 3. 加载模型 =====
print("2. 正在加载模型...")

start = time.perf_counter()

model = AutoModel(
    model=model_dir,
    device="cpu"
)

print("模型加载成功！")
print(f"加载耗时：{time.perf_counter() - start:.2f} 秒")


# ===== 4. 检查录音文件 =====
if not audio_path.is_file():
    raise FileNotFoundError(
        f"找不到测试录音：{audio_path}"
    )


# ===== 5. 进行语音识别 =====
print("3. 正在识别语音...")

start = time.perf_counter()

result = model.generate(
    input=str(audio_path),
    language="zh",
    use_itn=True,
    batch_size_s=60
)

elapsed = time.perf_counter() - start


# ===== 6. 获取识别文字 =====
text = rich_transcription_postprocess(
    result[0]["text"]
)

print("识别结果：", text)
print(f"识别耗时：{elapsed:.2f} 秒")
