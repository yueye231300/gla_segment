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
from scipy.spatial import distance
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

def analyze_all_lakes(mask, pixel_size=1, min_area=100):
    """分析所有冰湖
    Args:
        mask: 二值化掩码图像
        pixel_size: 每个像素代表的实际大小（米）
        min_area: 最小面积阈值，过滤掉太小的区域
    Returns:
        list: 包含所有冰湖的指标
    """
    # 确保掩码是二值图像
    binary_mask = (mask > 128).astype(np.uint8)
    
    # 找到所有轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lakes = []
    for idx, contour in enumerate(contours, 1):
        # 计算面积并过滤小区域
        area_pixels = cv2.contourArea(contour)
        if area_pixels < min_area:
            continue
            
        # 计算实际面积
        area = area_pixels * (pixel_size ** 2)
        
        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 计算长度和宽度
        width = rect[1][0] * pixel_size
        length = rect[1][1] * pixel_size
        
        # 确保长度大于宽度
        if width > length:
            width, length = length, width
        
        # 计算长宽比
        aspect_ratio = length / width if width > 0 else 0
        
        # 计算周长
        perimeter = cv2.arcLength(contour, True) * pixel_size
        
        # 计算质心
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
        
        lakes.append({
            'id': idx,
            'area': area,
            'length': length,
            'width': width,
            'aspect_ratio': aspect_ratio,
            'perimeter': perimeter,
            'contour': contour,
            'box': box,
            'centroid': (cx, cy)
        })
    
    # 按面积降序排序
    lakes.sort(key=lambda x: x['area'], reverse=True)
    return lakes

def draw_lakes_visualization(image, lakes):
    """在图像上绘制所有冰湖的标识和测量结果"""
    # 创建图像副本
    result = image.copy()
    if isinstance(result, np.ndarray) and len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    # 创建不同颜色的循环
    colors = [
        (255, 0, 0),   # 红
        (0, 255, 0),   # 绿
        (0, 0, 255),   # 蓝
        (255, 255, 0), # 黄
        (255, 0, 255), # 紫
        (0, 255, 255), # 青
    ]
    
    for i, lake in enumerate(lakes):
        color = colors[i % len(colors)]
        
        # 绘制轮廓
        cv2.drawContours(result, [lake['contour']], 0, color, 2)
        
        # 绘制湖泊编号
        cv2.putText(
            result,
            f"#{lake['id']}", 
            lake['centroid'],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            1
        )
    
    return result
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
            # 添加使用说明
        st.markdown("---")
        with st.expander("使用说明"):
            st.markdown("""
            ### 使用步骤
            1. 在左侧选择要使用的分割模型（UNet或DeepLabV3+）
            2. 上传待分析的图片或使用示例图片
            3. 上传图片可以使用天地图进行下载，下载之后可以得到像素对应的实际距离，便于得到准确的冰湖面积和周长
            4. 系统将自动进行分割并显示结果
            5. 对所有冰湖进行分割，并给出大于最小面积的冰湖的详细信息，包括面积，周长，长度，宽度和长宽比
            6. 可以下载分割掩码、叠加效果和冰湖测量结果
        
            ### 注意事项
            - 支持的图片格式：JPG、JPEG、PNG
            - 建议上传清晰的冰湖图片以获得最佳效果
            - 处理大图片时请耐心等待
            - 不同模型可能适用于不同类型的冰湖，可以尝试不同模型以获得更好的效果
        
            ### 模型说明
            - **UNet**: 经典的图像分割模型，适合边缘细节的提取
            - **DeepLabV3+**: 改进的分割模型，具有更好的语义理解能力
            """)
    with col2:
        st.subheader("分割结果")
        
        if models_loaded and 'image' in locals():
            model = unet_model if model_type == "UNet" else deeplabv3_model
            
            # 添加最小面积阈值设置
            min_area = st.slider(
                "最小冰湖面积 (像素)",
                min_value=10,
                max_value=1000,
                value=100,
                help="过滤掉小于此面积的区域"
            )
            
            # 添加像素尺寸设置
            pixel_size = st.number_input(
                "像素实际大小（米）",
                min_value=0.1,
                value=1.0,
                help="设置每个像素代表的实际距离（米）"
            )
            
            with st.spinner('正在进行图像分割和分析...'):
                mask = predict_mask(model, image, device)
                overlay = overlay_mask(image, mask)
                
                # 分析所有冰湖
                lakes = analyze_all_lakes(mask, pixel_size, min_area)
                
                # 绘制可视化结果
                lakes_viz = draw_lakes_visualization(np.array(image), lakes)
            
            # 显示结果
            tabs = st.tabs(["原始图片", "分割掩码", "叠加效果", "冰湖标识"])
            
            with tabs[0]:
                st.image(image, caption="原始图片", use_container_width=True)
            
            with tabs[1]:
                st.image(mask, caption="分割掩码", use_container_width=True)
            
            with tabs[2]:
                st.image(overlay, caption="叠加效果", use_container_width=True)
                
            with tabs[3]:
                st.image(lakes_viz, caption="冰湖标识", use_container_width=True)
            
            # 显示所有冰湖的详细信息
            st.markdown("### 所有冰湖测量结果")
            
            # 创建数据表格
            lakes_data = []
            for lake in lakes:
                lakes_data.append({
                    "冰湖编号": f"#{lake['id']}",
                    "面积 (㎡)": f"{lake['area']:.2f}",
                    "周长 (m)": f"{lake['perimeter']:.2f}",
                    "长度 (m)": f"{lake['length']:.2f}",
                    "宽度 (m)": f"{lake['width']:.2f}",
                    "长宽比": f"{lake['aspect_ratio']:.2f}"
                })
            
            # 显示表格
            st.dataframe(
                lakes_data,
                width=None,
                height=None,
                use_container_width=True
            )
            
            # 添加数据下载按钮
            import pandas as pd
            csv = pd.DataFrame(lakes_data).to_csv(index=False)
            st.download_button(
                "下载测量结果",
                csv,
                "lakes_measurements.csv",
                "text/csv",
                key='download-csv'
            )
            
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

            # 添加面积分布可视化
            if len(lakes) > 1:
                st.markdown("### 冰湖面积分布")
                import plotly.express as px
                
                areas_df = pd.DataFrame([
                    {"冰湖编号": f"#{lake['id']}", "面积": lake['area']}
                    for lake in lakes
                ])
                
                fig = px.bar(
                    areas_df,
                    x="冰湖编号",
                    y="面积",
                    title="冰湖面积分布"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # st.markdown("### 形状分析")
            # shape_metrics = []
            # for lake in lakes:
            #     # 计算圆形度
            #     circularity = 4 * np.pi * lake['area'] / (lake['perimeter'] ** 2)
            #     shape_metrics.append({
            #         "冰湖编号": f"#{lake['id']}",
            #         "圆形度": f"{circularity:.3f}"
            #     })
            
            # st.dataframe(
            #     shape_metrics,
            #     width=None,
            #     height=None,
            #     use_container_width=True
            # )
    
if __name__ == "__main__":
    main()