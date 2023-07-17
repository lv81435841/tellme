"""
 作者 lgf
 日期 2023/5/3
"""
import whisper

model = whisper.load_model("medium")
result = model.transcribe("1.m4a")
print(result["text"])