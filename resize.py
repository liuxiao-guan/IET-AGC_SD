import os
from PIL import Image
import torchvision.transforms as transforms

# 输入文件夹和输出文件夹
input_folder = '/root/autodl-tmp/logs/Projects/DCR/ret_plots/DCR/laion_frozentext/_instancelevel_blip_nodup_bs4_gpu4/similar_train'  # 原图片文件夹路径
output_folder = '/root/autodl-tmp/logs/Projects/DCR/ret_plots/DCR/laion_frozentext/_instancelevel_blip_nodup_bs4_gpu4/similar_train_resize'  # 新图片文件夹路径

# 创建输出文件夹（如果不存在的话）
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中的所有文件
image_files = [f for f in os.listdir(input_folder) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

# 定义转化方式：裁剪和填充
transform = transforms.Compose([
    transforms.Resize(256),  # 保持比例调整为最长边为256，短边按比例缩放
    transforms.CenterCrop(256),  # 从中心裁剪为 256x256
    transforms.ToTensor()  # 将图片转换为 Tensor
])

# 遍历每个图片文件
for image_file in image_files:
    # 获取图片的完整路径
    image_path = os.path.join(input_folder, image_file)
    
    # 打开图片
    img = Image.open(image_path)
    
    # 应用变换
    img_transformed = transform(img)
    
    # 将 Tensor 转回图片
    img_transformed = transforms.ToPILImage()(img_transformed)
    
    # 保存变换后的图片
    output_image_path = os.path.join(output_folder, image_file)
    img_transformed.save(output_image_path)

    print(f'Image {image_file} resized and saved to {output_image_path}')