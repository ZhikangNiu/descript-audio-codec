import pandas as pd
import argparse

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Calculate mean values of specific columns in a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    args = parser.parse_args()
    
    # 读取CSV数据
    data = pd.read_csv(f"{args.csv_path}/metrics.csv")
    
    # 计算每一列指标的均值（除去name列）
    mean_values = data.drop(columns=['name']).mean()
    
    # 输出结果
    print("Mean values for each column (excluding 'name'):")
    print(mean_values)

if __name__ == "__main__":
    main()
