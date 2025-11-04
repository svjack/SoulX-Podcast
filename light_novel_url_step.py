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

### colab

import os
import requests
import pandas as pd
import json
from urllib.parse import unquote
from pathlib import Path

def download_epub_with_metadata(df, output_folder="output"):
    """
    下载DataFrame中所有URL指向的EPUB文件，并生成对应的JSON元数据文件
    
    参数:
        df: DataFrame，包含url列和其他元数据列
        output_folder: 输出文件夹路径
    """
    # 创建输出文件夹
    Path(output_folder).mkdir(exist_ok=True)
    
    # 记录成功和失败的项目
    success_count = 0
    error_count = 0
    error_list = []
    
    for index, row in df.iterrows():
        try:
            # 从URL中提取文件名
            epub_url = row['url']
            filename = extract_filename_from_url(epub_url)
            
            if not filename.endswith('.epub'):
                filename += '.epub'
            
            # 设置文件路径
            epub_path = os.path.join(output_folder, filename)
            json_path = os.path.join(output_folder, filename.replace('.epub', '.json'))
            
            print(f"正在处理: {filename}")
            
            # 下载EPUB文件
            if download_epub_file(epub_url, epub_path):
                # 创建元数据JSON（排除urls和url列）
                metadata = create_metadata_json(row, df.columns)
                
                # 保存JSON文件
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                print(f"✓ 成功下载: {filename}")
                success_count += 1
            else:
                error_count += 1
                error_list.append(filename)
                print(f"✗ 下载失败: {filename}")
                
        except Exception as e:
            error_count += 1
            error_list.append(f"{filename} - 错误: {str(e)}")
            print(f"✗ 处理失败 {filename}: {str(e)}")
    
    # 输出总结报告
    print(f"\n=== 处理完成 ===")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {error_count} 个文件")
    if error_list:
        print("失败列表:")
        for error in error_list:
            print(f"  - {error}")

def extract_filename_from_url(url):
    """
    从URL中提取文件名，处理URL编码和特殊字符
    """
    # 解码URL编码的字符
    decoded_url = unquote(url)
    
    # 从URL中提取文件名部分
    filename = decoded_url.split('/')[-1]
    
    # 清理文件名中的特殊字符（保留中文、英文、数字、空格、横杠、下划线）
    clean_filename = re.sub(r'[^\w\s\u4e00-\u9fff\-\.]', '', filename)
    
    return clean_filename.strip()

def download_epub_file(url, save_path):
    """
    下载EPUB文件到指定路径 [1](@ref)
    
    返回:
        bool: 下载是否成功
    """
    try:
        # 设置请求头，模拟浏览器行为
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # 发送GET请求
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        # 检查文件大小
        file_size = int(response.headers.get('content-length', 0))
        print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
        
        # 分块下载大文件
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        
        # 验证文件是否成功下载
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True
        else:
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"下载请求失败: {e}")
        return False
    except Exception as e:
        print(f"下载过程出错: {e}")
        return False

def create_metadata_json(row, columns):
    """
    创建元数据JSON，排除urls和url列
    """
    metadata = {}
    
    for col in columns:
        if col not in ['urls', 'url']:
            # 处理可能存在的NaN值
            value = row[col]
            if pd.isna(value):
                metadata[col] = None
            else:
                metadata[col] = value
    
    # 添加处理时间戳
    metadata['download_timestamp'] = pd.Timestamp.now().isoformat()
    
    return metadata

# 添加重试机制的增强版本
def download_with_retry(url, save_path, max_retries=3):
    """
    带重试机制的文件下载函数
    """
    for attempt in range(max_retries):
        try:
            if download_epub_file(url, save_path):
                return True
            else:
                print(f"第 {attempt + 1} 次尝试失败，重试...")
        except Exception as e:
            print(f"第 {attempt + 1} 次尝试失败: {e}")
    
    return False

# 主程序执行
if __name__ == "__main__":
    import re  # 确保导入re模块
    
    # 加载数据
    Light_Novel_2025_11_base_df = pd.read_csv("Light_Novel_2025_11_base.csv")
    Light_Novel_2025_11_base_df["urls"] = Light_Novel_2025_11_base_df["urls"].map(eval)
    Light_Novel_2025_11_base_df["url"] = Light_Novel_2025_11_base_df["urls"].map(lambda x: x[0])
    
    # 执行下载
    download_epub_with_metadata(Light_Novel_2025_11_base_df, "epub_output")
