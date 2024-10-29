import os
import numpy as np
import pandas as pd

# 文件路径
workdir = r'E:\数据\黄昭景\数据'
label_file = r'E:\数据\黄昭景\标签\12色正交表.xls'

# 获取需要处理的文件列表
file_list = [os.path.join(workdir, file) for file in os.listdir(workdir) if file.endswith('.xls')]

# 初始化空列表，用于存储所有数据
all_samples = []

# 读取所有文件并存储数据
for file in file_list:
    # 提取数据标号，假设文件名格式为 D-index-1.wls.xls
    index = int(os.path.basename(file).split('-')[1])

    # 读取Excel文件
    df = pd.read_excel(file, sheet_name=0)

    # 提取第 5 列的数据，从第 8 行开始，并且重置索引
    # 检查DataFrame的形状是否允许索引
    if df.shape[0] > 7 and df.shape[1] > 4:
        column_cm = df.iloc[7:, 4].reset_index(drop=True).values.astype(float)

        # 将数据存入列表，每个元素是一个字典，代表一个样本
        all_samples.append({
            'Index': index,
            'Data': column_cm,
        })
    else:
        print(f"文件 {file} 中数据行数或列数不足，无法提取数据")

print("Excel 文件数据处理完毕，共处理 {} 个文件".format(len(file_list)))

# 将数据转换为DataFrame
all_data = pd.DataFrame(all_samples)

# # 读取标签数据
df_label = pd.read_excel(label_file)

# 合并标签数据
all_data = pd.merge(all_data, df_label[['Index', 'Fe2+']], on='Index', how='left')

# 将Data列展开为多列（这里假设Data列展开后有多列，修改为你实际的列数）
expanded_data = pd.DataFrame(all_data['Data'].tolist())

# 合并Index、Ag和展开后的Data列
all_data = pd.concat([all_data[['Index', 'Fe2+']], expanded_data], axis=1)

# 将处理后的结果保存到 CSV 文件
all_data.to_csv(r"dataset\Fe2+-data.csv", index=False)

print("数据处理完毕，结果已保存到 data.csv 文件")
