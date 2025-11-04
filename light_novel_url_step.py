'''
url 

https://mojimoon.github.io/wenku8/

html

as

novel.html
'''

import pandas as pd
from bs4 import BeautifulSoup
import requests

def html_table_to_dataframe(html_file_path):
    """
    将本地HTML文件中的表格转换为Pandas DataFrame
    保留年更列中的所有内容，包括超链接
    
    参数:
        html_file_path (str): HTML文件路径
        
    返回:
        pd.DataFrame: 包含表格数据的DataFrame
    """
    # 读取本地HTML文件
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 找到表格
    table = soup.find('table')
    
    # 提取表头
    headers = []
    header_row = table.find('tr')
    for th in header_row.find_all('th'):
        headers.append(th.get_text(strip=True))
    
    # 提取表格数据
    data = []
    for row in table.find_all('tr')[1:]:  # 跳过表头行
        row_data = []
        cells = row.find_all(['td', 'th'])
        
        for i, cell in enumerate(cells):
            # 对于"年更"列（索引4），保留所有内容包括HTML
            if i == 4:  # "年更"是第5列（索引从0开始）
                # 提取年更列中的所有链接和文本
                links = cell.find_all('a')
                if links:
                    link_content = []
                    for link in links:
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        link_content.append(f"{text}({href})")
                    row_data.append(' | '.join(link_content))
                else:
                    row_data.append(cell.get_text(strip=True))
            else:
                # 其他列只提取文本
                row_data.append(cell.get_text(strip=True))
        
        data.append(row_data)
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    return df

def simple_html_to_dataframe(html_file_path):
    """
    使用pandas的read_html函数直接读取表格
    注意：这种方法可能无法完美保留年更列的超链接
    
    参数:
        html_file_path (str): HTML文件路径
        
    返回:
        pd.DataFrame: 包含表格数据的DataFrame
    """
    try:
        # 读取所有表格
        tables = pd.read_html(html_file_path)
        # 假设目标表格是第一个
        df = tables[0]
        return df
    except Exception as e:
        print(f"使用pandas read_html失败: {e}")
        return None

def clean_dataframe(df):
    """
    对DataFrame进行数据清洗
    
    参数:
        df (pd.DataFrame): 原始DataFrame
        
    返回:
        pd.DataFrame: 清洗后的DataFrame
    """
    # 去除空行
    df_cleaned = df.dropna(how='all')
    # 重置索引
    df_cleaned = df_cleaned.reset_index(drop=True)
    # 处理空值
    df_cleaned = df_cleaned.fillna('')
    
    return df_cleaned

# 主程序
if __name__ == "__main__":
    # 使用方法
    file_path = "novel.html"  # 替换为您的HTML文件路径
    
    # 方法1: 使用BeautifulSoup（推荐，保留超链接）
    print("=" * 50)
    print("方法1: 使用BeautifulSoup解析（保留超链接）")
    print("=" * 50)
    
    df = html_table_to_dataframe(file_path)
    
    # 显示DataFrame
    print("转换后的DataFrame:")
    print(df.head())
    
    # 显示DataFrame信息
    print("\nDataFrame信息:")
    print(df.info())
    
    # 显示年更列的详细内容
    print("\n年更列内容示例:")
    for i, content in enumerate(df['年更'].head()):
        print(f"行 {i}: {content}")
    
    # 方法2: 尝试简化方法
    print("\n" + "=" * 50)
    print("方法2: 使用pandas read_html（简化版）")
    print("=" * 50)
    
    df_simple = simple_html_to_dataframe(file_path)
    if df_simple is not None:
        print("使用pandas read_html的结果:")
        print(df_simple.head())
    else:
        print("简化方法失败，使用BeautifulSoup方法")
    
    # 数据清洗和处理
    print("\n" + "=" * 50)
    print("数据清洗和处理")
    print("=" * 50)
    
    df_cleaned = clean_dataframe(df)
    print("清洗后的DataFrame:")
    print(df_cleaned.head())



# 测试您提供的示例文本
example_text = '下载(https://raw.githubusercontent.com/ixinzhi/lightnovel-2024/master/怪人的沙拉碗 - 平坂读 - 20240902.epub) | 镜像(https://ghfast.top/https://raw.githubusercontent.com/ixinzhi/lightnovel-2024/master/怪人的沙拉碗 - 平坂读 - 20240902.epub)'

import re

def extract_epub_urls_reliable(text):
    """
    最可靠的EPUB链接提取方法
    """
    # 方法1：匹配括号内的完整URL（针对您的特定格式）
    pattern1 = r'\((https?://[^\)]+?\.epub)\)'
    
    # 方法2：通用的URL匹配
    pattern2 = r'https?://[^\s\'\"]+?\.epub'
    
    urls = []
    
    # 尝试第一种模式
    matches1 = re.findall(pattern1, text, re.IGNORECASE)
    urls.extend(matches1)
    
    # 如果第一种模式没找到，尝试第二种
    if not urls:
        matches2 = re.findall(pattern2, text, re.IGNORECASE)
        urls.extend(matches2)
    
    return urls

# 最终测试
final_urls = extract_epub_urls_reliable(example_text)
print("\n最终提取结果:")
for i, url in enumerate(final_urls, 1):
    print(f"{i}. {url}")

novel_base_df = df_cleaned[["标题", "作者", "年更"]].copy()
novel_base_df

novel_base_df["urls"] = novel_base_df["年更"].map(extract_epub_urls_reliable)

novel_base_df[
novel_base_df["urls"].map(bool)
].reset_index().iloc[:, 1:].to_csv("Light_Novel_2025_11_base.csv", index = False)