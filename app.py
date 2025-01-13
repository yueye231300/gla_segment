import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io
from huggingface_hub import hf_hub_download
from models import (
    UNet, 
    DeepLabV3Plus,
    predict_mask
)

# 页面配置
st.set_page_config(
    page_title="冰湖分割系统",
    page_icon="🌊",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin-top: 1em;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .result-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    .header-container {
        text-align: center;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """加载模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 加载UNet模型
        unet_model = UNet()
        model_path_1 = hf_hub_download(
            repo_id = "yuyue231300/glacial_lake_model",
            filename = 'best_model_unet.pth'
        )
        unet_state = torch.load(model_path_1, map_location=device)
        if 'model_state_dict' in unet_state:
            unet_model.load_state_dict(unet_state['model_state_dict'])
        else:
            unet_model.load_state_dict(unet_state)
        unet_model.to(device)
        unet_model.eval()
        
        # 加载DeepLabV3+模型
        deeplabv3_model = DeepLabV3Plus()
        model_path = hf_hub_download(
            repo_id="yuyue231300/glacial_lake_model",
            filename="best_model_deeplabv3plus.pth"  # 你的模型文件名
        )
        deeplabv3_state = torch.load(model_path, map_location=device)
        if 'model_state_dict' in deeplabv3_state:
            deeplabv3_model.load_state_dict(deeplabv3_state['model_state_dict'])
        else:
            deeplabv3_model.load_state_dict(deeplabv3_state)
        deeplabv3_model.to(device)
        deeplabv3_model.eval()
        
        return unet_model, deeplabv3_model, device
        
    except Exception as e:
        st.error(f"模型加载错误: {str(e)}")
        raise e

@st.cache_data
def load_sample_image():
    """加载示例图片"""
    return Image.open('demo.png')

def overlay_mask(image, mask, alpha=0.5):
    """将分割掩码叠加到原始图像上"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 128] = [0, 255, 0]
    
    overlay = cv2.addWeighted(image, 1, mask_colored, alpha, 0)
    return overlay

def calculate_metrics(mask, threshold=128):
    """计算分割区域的指标"""
    # 确保掩码是二值图像
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # 计算总像素数和冰湖像素数
    total_pixels = mask.size
    ice_pixels = np.sum(binary_mask)
    ice_percentage = (ice_pixels / total_pixels) * 100
    
    return {
        'total_pixels': total_pixels,
        'ice_pixels': ice_pixels,
        'ice_percentage': ice_percentage
    }

def main():
    st.markdown("""
        <div class="header-container">
            <h1>🌊 冰湖分割系统</h1>
            <p>使用深度学习实现冰湖区域自动分割</p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        # 加载模型
        unet_model, deeplabv3_model, device = load_models()
        models_loaded = True
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        models_loaded = False
        return
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    # 左侧列：模型选择和图片上传
    with col1:
        st.subheader("配置")
        
        # 模型选择
        model_type = st.radio(
            "选择模型",
            ("UNet", "DeepLabV3+"),
            help="选择要使用的分割模型"
        )
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "上传图片",
            type=['jpg', 'jpeg', 'png'],
            help="支持jpg、jpeg、png格式的图片"
        )
        
        # 示例图片按钮
        if st.button("使用示例图片"):
            st.session_state['use_sample'] = True
            
        # 获取要处理的图片
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # 转换为RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
        elif st.session_state.get('use_sample', False):
            image = load_sample_image()
        else:
            st.info("请上传图片或使用示例图片")
            return
            
        # 显示原始图片信息
        st.markdown("### 图片信息")
        st.write(f"原始尺寸: {image.size[0]} × {image.size[1]} 像素")
    
    # 右侧列：结果显示
    with col2:
        st.subheader("分割结果")
        
        if models_loaded and 'image' in locals():
            # 选择模型
            model = unet_model if model_type == "UNet" else deeplabv3_model
            
            # 处理图片
            with st.spinner('正在进行图像分割...'):
                mask = predict_mask(model, image, device)
                overlay = overlay_mask(image, mask)
            
            # 显示结果
            tabs = st.tabs(["原始图片", "分割掩码", "叠加效果"])
            
            with tabs[0]:
                st.image(image, caption="原始图片", use_container_width=True)
            
            with tabs[1]:
                st.image(mask, caption="分割掩码", use_container_width=True)
            
            with tabs[2]:
                st.image(overlay, caption="叠加效果", use_container_width=True)
            
            # 计算和显示指标
            metrics = calculate_metrics(mask)
            st.markdown("### 分析结果")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("总像素数", f"{metrics['total_pixels']:,}")
            with col4:
                st.metric("冰湖像素数", f"{metrics['ice_pixels']:,}")
            with col5:
                st.metric("冰湖占比", f"{metrics['ice_percentage']:.2f}%")
            
            # 下载按钮
            st.markdown("### 导出结果")
            col6, col7 = st.columns(2)
            
            with col6:
                # 保存掩码为bytes
                mask_bytes = io.BytesIO()
                Image.fromarray(mask).save(mask_bytes, format='PNG')
                st.download_button(
                    label="下载掩码",
                    data=mask_bytes.getvalue(),
                    file_name="segmentation_mask.png",
                    mime="image/png"
                )
            
            with col7:
                # 保存叠加效果为bytes
                overlay_bytes = io.BytesIO()
                Image.fromarray(overlay).save(overlay_bytes, format='PNG')
                st.download_button(
                    label="下载叠加效果",
                    data=overlay_bytes.getvalue(),
                    file_name="overlay_result.png",
                    mime="image/png"
                )
    
    # 添加使用说明
    st.markdown("---")
    with st.expander("使用说明"):
        st.markdown("""
        ### 使用步骤
        1. 在左侧选择要使用的分割模型（UNet或DeepLabV3+）
        2. 上传待分析的图片或使用示例图片
        3. 系统将自动进行分割并显示结果
        4. 可以下载分割掩码或叠加效果图
        
        ### 注意事项
        - 支持的图片格式：JPG、JPEG、PNG
        - 建议上传清晰的冰湖图片以获得最佳效果
        - 处理大图片时请耐心等待
        
        ### 模型说明
        - **UNet**: 经典的图像分割模型，适合边缘细节的提取
        - **DeepLabV3+**: 改进的分割模型，具有更好的语义理解能力
        """)

if __name__ == "__main__":
    main()