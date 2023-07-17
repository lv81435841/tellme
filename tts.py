"""
 作者 lgf
 日期 2023/4/28
"""
from bark import SAMPLE_RATE, generate_audio, preload_models
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     我要你扮演一个小学老师,回答学生给出的问题，必须和蔼可亲，回答要精确并适合学生理解，一次性回答完
"""
audio_array = generate_audio(text_prompt)

# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)