import os
import sys
import torch

# ========================================================
# 1. 兼容性补丁：修复 torchvision 找不到 functional_tensor 的问题
# ========================================================
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as F
        sys.modules['torchvision.transforms.functional_tensor'] = F
    except ImportError:
        pass

import cv2
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def upscale_video_m4(input_path, output_path, target_width=1280, target_height=704):
    if not os.path.exists(input_path):
        print(f"❌ 找不到文件: {input_path}")
        return

    # --- 模型参数配置 ---
    # 使用经典 RealESRGAN_x4plus 模型
    model_name = 'RealESRGAN_x4plus'
    # 显式指定模型下载地址，防止库内部报 NoneType 错误
    model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    
    # 初始化模型架构 (RRDBNet)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # 强制使用 M4 GPU 加速
    device = torch.device('mps') 
    print(f"🚀 已激活 M4 GPU 加速 (MPS)")

    # --- 初始化超分器 ---
    try:
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_url, # 这里传 URL，库会自动下载到 weights 文件夹
            model=model,
            tile=400,            # 分块处理，防止 4K 等超大图爆内存
            tile_pad=10,
            pre_pad=0,
            half=True,           # M4 支持 FP16 硬件加速，速度翻倍
            device=device
        )
    except Exception as e:
        print(f"❌ 初始化超分器失败: {e}")
        return

    # --- 读取视频 ---
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置临时输出
    temp_avi = "temp_upscaled.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_avi, fourcc, fps, (target_width, target_height))

    print(f"🎬 开始高清处理: {total_frames} 帧")

    count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # AI 超分处理 (核心计算)
            # outscale=4 表示 AI 内部放大 4 倍
            output, _ = upsampler.enhance(frame, outscale=4)
            
            # 使用高画质 Lanczos 算法缩放到你要求的 1280x704
            final_frame = cv2.resize(output, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            
            out.write(final_frame)
            count += 1
            if count % 5 == 0:
                print(f"⏳ 进度: {count}/{total_frames} 帧", end='\r')
            
    except Exception as e:
        print(f"\n❌ 运行中出错: {e}")
    finally:
        cap.release()
        out.release()

    # --- 最终压制 (M4 硬件编码加速) ---
    print(f"\n📦 正在使用 M4 媒体引擎压制最终视频...")
    # 这里使用 h264_videotoolbox 是苹果芯片最快的压制方式
    final_cmd = f'ffmpeg -y -i "{temp_avi}" -c:v h264_videotoolbox -b:v 8000k "{output_path}"'
    
    os.system(final_cmd)
    
    if os.path.exists(temp_avi):
        os.remove(temp_avi)
        
    print(f"✨ 高清放大完成！输出至: {output_path}")

if __name__ == "__main__":
    # 执行脚本
    upscale_video_m4("5_no_wm.mp4", "5_hd.mp4")