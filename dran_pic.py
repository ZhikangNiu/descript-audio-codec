import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_data_from_folder(folder_path):
    """
    遍历文件夹下所有的 npy 文件，加载并拼接成一个一维数组。
    """
    # 获取指定文件夹下所有 .npy 文件路径
    file_list = list(Path(folder_path).rglob("*.npy"))
    all_data = []
    
    for file in file_list:
        try:
            # 加载 npy 文件（假设每个文件中的数组维度可以不同，这里统一 flatten 成一维）
            data = np.load(file)
            data = data.flatten()
            all_data.append(data)
        except Exception as e:
            print(f"加载文件 {file} 时发生错误: {e}")
    
    if all_data:
        # 将所有数组拼接成一个一维数组
        all_data = np.concatenate(all_data, axis=0)
    else:
        all_data = np.array([])
    return all_data

def plot_distribution(data, title, bins=100, save_path=None):
    """
    绘制数据直方图，显示数据的概率密度分布。
    画布尺寸放大且高清，网格以1为一格显示。
    """
    # 设置画布大小及分辨率
    plt.figure(figsize=(16, 12), dpi=150)
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='b')
    plt.title(title, fontsize=20)
    plt.xlabel('number', fontsize=16)
    plt.ylabel('percent', fontsize=16)
    
    # 开启网格，并设置网格样式
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 设置坐标轴刻度，以1为间隔
    ax = plt.gca()
    
    # X轴刻度
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.arange(np.floor(xmin), np.ceil(xmax)+1, 1))
    
    # Y轴刻度
    ymin, ymax = ax.get_ylim()
    ax.set_yticks(np.arange(np.floor(ymin), np.ceil(ymax)+1, 1))
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图像已保存至 {save_path}")
    plt.show()

if __name__ == '__main__':
    # 指定 VAE 数据所在的文件夹路径
    vae_folder = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/LibriTTS/30hz_feat/train-clean-100'
    # 如果有 Mel 数据，可类似设置 mel_folder
    mel_folder = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/LibriTTS/vocos/train-clean-100"

    # 加载数据
    # vae_data = load_data_from_folder(vae_folder)
    mel_data = load_data_from_folder(mel_folder)
    
    # 输出基本统计信息，方便调试和了解数据范围
    # print("VAE Data")
    # print(f"sum: {vae_data.size}")
    # print(f"min: {vae_data.min()}, max: {vae_data.max()}, mean: {vae_data.mean()}")
    # 例如：最小值：-22.649843215942383, 最大值：24.014253616333008, 均值：0.020335979759693146
    
    # 绘制直方图并保存图像
    # plot_distribution(vae_data, 'VAE', bins=100, save_path='vae_distribution.png')
    # 如有需要，类似处理 Mel 数据：
    plot_distribution(mel_data, 'Mel', bins=100, save_path='mel_distribution.png')
