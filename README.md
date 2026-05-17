代码说明：


很多时候我们如果不充会员，用云端网站做出的视频往往都带有水印，并且长宽也无法自己定义，用我下面的这2份代码，
rmwatermark.py 去除视频水印
hdvideo.py 高清放大视频
可以实现水印消除和高清放大，纯本地运行。
去水印使用OpenCV的算法，高清放大使用Real-ESRGAN，第一次运行时只需要下载大约60M的模型即可。
我使用mac m4跑的，如果用别的平台，代码整体逻辑都不变，哪里都能运行。
先安装下面的依赖，除了python的包以外，还需要安装ffmpeg。
另外，处理的时候为了提高速度，我的视频没有声音，如果你的视频有声音的话，建议先用剪映等工具把音频去掉，处理好后再把声音合并上去。
如果因为平台不同代码运行报错，就把代码和错误信息随便丢给一个AI，修改下就行了，应该不会有大问题。

# 安装基础 AI 库
pip install torch torchvision
# 安装 Real-ESRGAN 处理库
pip install realesrgan
# 安装 opencv
pip install opencv-python
pip install basicsr
pip install numpy
brew install ffmpeg
