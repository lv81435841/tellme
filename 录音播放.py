"""
 作者 lgf
 日期 2023/4/26
"""
import wave
import pyaudio
import numpy as np

record_seconds = 10  # 需要录制的时间
rate = 16000  # 录音时的采样率
wave_output_filename = 'listen.wav'  # 保存的文件名
chunk = 1024  # 每个缓冲区的帧数
format = pyaudio.paInt16  # 采样大小和格式
channels = 1  # 通道数
pa = pyaudio.PyAudio()  # 初始化端口音频系统资源
stream = pa.open(
    format=format,
    channels=channels,
    rate=rate,
    input=True,  # 指定当前为输入流
    frames_per_buffer=chunk,
)  # 开启流
frames = []  # 音频帧列表
min_decibels = 99999  # 音频最小分贝
max_decibels = 0  # 音频最大分贝
for i in range(0, int(rate / chunk * record_seconds)):
    data = stream.read(chunk)
    frames.append(data)  # 开始录音
    # 以下代码用于监测音频分贝
    audio_data = np.fromstring(data, dtype=np.short)
    temp = np.max(audio_data)
    if max_decibels < int(temp):
        max_decibels = int(temp)  # 更新最大分贝
    if min_decibels > int(temp):
        min_decibels = int(temp)  # 更新最小分贝
stream.stop_stream()  # 停止流
stream.close()  # 关闭流
pa.terminate()  # 释放端口音频系统资源
wf = wave.open(wave_output_filename, 'wb')  # 生成 wav_write 对象
wf.setnchannels(channels)  # 设置对象通道数
wf.setsampwidth(pa.get_sample_size(format))  # 设置对象采样字节长度
wf.setframerate(rate)  # 设置对象采样频率
wf.writeframes(b''.join(frames))  # 写入音频帧并确保正确性
wf.close()  # 关闭 wav_write 对象

print(f'最大分贝(原始值) = {max_decibels}')
print(f'最小分贝(原始值) = {min_decibels}')



filename = 'listen.wav'

# Set chunk size of 1024 samples per data frame
chunk = 1024

# Open the soaudio/sound file
af = wave.open(filename,'rb')

# Create an interface to PortAudio
pa = pyaudio.PyAudio()

# Open a .Stream object to write the WAV file
# 'output = True' indicates that the
# sound will be played rather than
# recorded and opposite can be used for recording
stream = pa.open(format=pa.get_format_from_width(af.getsampwidth()),
                 channels=af.getnchannels(),
                 rate=af.getframerate(),
                 output=True)

# Read data in chunks
rd_data = af.readframes(chunk)

# Play the sound by writing the audio
# data to the Stream using while loop
while rd_data != '':
    stream.write(rd_data)
    rd_data = af.readframes(chunk)

# Close and terminate the stream
stream.stop_stream()
stream.close()
pa.terminate()
