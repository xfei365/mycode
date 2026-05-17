import cv2
import numpy as np
import subprocess
import os

def remove_watermark_silent(input_video, output_video, x, y, w, h):
    """
    专门针对无声视频的去水印脚本
    x, y: 水印左上角坐标
    w, h: 水印宽高
    """
    if not os.path.exists(input_video):
        print(f"❌ 找不到输入文件: {input_video}")
        return

    # 1. 打开视频
    cap = cv2.VideoCapture(input_video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if width == 0 or height == 0:
        print("❌ 视频读取失败，请检查文件。")
        return

    # 2. 准备中间临时文件
    temp_processed = "temp_raw_processed.mp4"
    # 使用 mp4v 编码生成临时无声视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_processed, fourcc, fps, (width, height))

    # 3. 准备遮罩 (Mask)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255

    print(f"🚀 M4 硬件加速启动...")
    print(f"🎬 正在处理无声视频: {input_video} ({total_frames} 帧)")

    # 4. 逐帧修复
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # AI 修复算法 (Telea 算法)
        # 效果好，且在 M4 的 CPU 上运行极快
        dst = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        out.write(dst)
        
        count += 1
        if count % 20 == 0 or count == total_frames:
            print(f"⏳ 进度: {count}/{total_frames} 帧", end='\r')

    cap.release()
    out.release()
    print("\n✅ 图像修复完成。")

    # 5. 最终转码 (调用 M4 硬件加速器 h264_videotoolbox)
    print("📦 正在进行最终硬件压制...")
    final_cmd = [
        'ffmpeg', '-y',
        '-i', temp_processed,
        '-c:v', 'h264_videotoolbox', # 调用苹果硅片硬编
        '-b:v', '6000k',             # 设置 6Mbps 高比特率确保清晰度
        output_video
    ]
    
    try:
        # 执行合成并静默日志
        subprocess.run(final_cmd, check=True, capture_output=True)
        print(f"✨ 处理成功！输出文件: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 最终合成失败: {e.stderr.decode()}")
    finally:
        # 清理
        if os.path.exists(temp_processed):
            os.remove(temp_processed)

# --- 运行参数 ---
# 请在此处修改你的坐标
# x: 水印距离左边多少像素
# y: 水印距离顶端多少像素
# w: 水印宽度
# h: 水印高度
remove_watermark_silent(
    input_video="EP01-6.mp4", 
    output_video="EP01-6_no_wm.mp4", 
    x=5, y=5, w=236, h=40 
)