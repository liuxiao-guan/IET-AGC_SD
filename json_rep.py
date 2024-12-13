


import json

# 定义旧文件夹路径和新文件夹路径
old_folder_path = "/fs/cml-projects/diffusion_rep/data/laion_10k_random/train/images_large/"
new_folder_path = "/root/autodl-tmp/laion_10k/train/images_large/"

# 读取JSON文件
with open('/root/autodl-tmp/laion_10k/laion_combined_captions.json', 'r') as file:
    data = json.load(file)


# 创建临时字典保存修改后的键值对
new_data = {}
for key, value in data.items():
    if key.startswith(old_folder_path):
        new_key = new_folder_path + key[len(old_folder_path):]
        new_data[new_key] = value
    else:
        new_data[key] = value

# 更新原始字典的内容
data.clear()
data.update(new_data)

# 将修改后的数据写回JSON文件
with open('/root/autodl-tmp/laion_10k/laion_combined_captions_modify.json', 'w') as file:
    json.dump(data, file, indent=4)

# import os

# # 图片文件夹路径
# image_folder = "/root/autodl-tmp/laion_10k/train/images_large/"

# # 获取图片文件名列表
# image_files = os.listdir(image_folder)

# # 提取文件名中的数字部分
# image_numbers = [int(file.split('.')[0]) for file in image_files]

# # 构造完整的数字列表
# full_number_list = list(range(10000))

# # 找出缺失的数字
# missing_numbers = [num for num in full_number_list if num not in image_numbers]

# # 输出缺失的数字
# print("缺失的图片数字：", missing_numbers)


