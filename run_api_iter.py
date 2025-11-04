'''
python run_api.py --model pretrained_models/SoulX-Podcast-1.7B

python run_api_iter.py --input_dir=No_6_Light_Novel_Xiang_IndexTTS2_Chapter1 --output_dir=No_6_Light_Novel_Xiang_SoulX_Podcast_Chapter1
'''

import requests
import time
import json
import re
import argparse
from pathlib import Path
from pydub import AudioSegment
import os

def clean_text(text):
    """
    清理单行文本，只保留中英文逗号、句号和问号，并去除所有空格
    保留的标点：,，.。？?
    """
    # 去除除了保留标点外的所有标点符号
    pattern = r'[^\w\s,，.。？?]'
    cleaned_text = re.sub(pattern, '', text)
    # 去除所有空格（包括普通空格、制表符等空白字符）
    cleaned_text = re.sub(r'\s+', '', cleaned_text)
    return cleaned_text

def split_text(text, lines_per_chunk=10):
    """
    先将文本按行分割，然后对每一行进行清理，最后每若干行合并为一个段落
    """
    # 首先按换行符分割原始文本得到各行
    lines = text.split('\n')
    # 对每一行进行清理并过滤空行
    cleaned_lines = [clean_text(line) for line in lines if line.strip()]
    
    chunks = []
    # 将清理后的行按指定数量合并成段落
    for i in range(0, len(cleaned_lines), lines_per_chunk):
        # 直接连接清理后的行，不添加任何空格
        chunk = ''.join(cleaned_lines[i:i+lines_per_chunk])
        # 替换多个连续句号为单个句号
        chunk = re.sub(r'。+', '。', chunk)
        # 最终确认无任何空格
        chunk = re.sub(r'\s+', '', chunk)
        chunks.append(chunk)
    
    return chunks

def generate_audio(api_url, prompt_audio_path, prompt_text, dialogue_text, seed=1988):
    """调用API生成音频"""
    # 在发送前再次确认文本无空格
    dialogue_text = re.sub(r'\s+', '', dialogue_text)
    
    files = {'prompt_audio': open(prompt_audio_path, 'rb')}
    data = {
        'prompt_texts': json.dumps([prompt_text]),
        'dialogue_text': dialogue_text,
        'seed': seed
    }
    
    try:
        response = requests.post(f"{api_url}/generate", files=files, data=data)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"✗ 请求失败: {e}")
        return None
    finally:
        files['prompt_audio'].close()

def process_txt_file(txt_path, output_dir, api_url, prompt_audio_path, prompt_text, lines_per_chunk=10):
    """处理单个txt文件"""
    print(f"\n处理文件: {txt_path}")
    
    # 读取文本文件
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            with open(txt_path, 'r', encoding='gbk') as f:
                text = f.read()
        except Exception as e:
            print(f"✗ 无法读取文件 {txt_path}: {e}")
            return
    
    print(f"原始文本长度: {len(text)} 字符")
    
    # 分割文本为指定行数一段
    text_chunks = split_text(text, lines_per_chunk)
    print(f"分割为 {len(text_chunks)} 个段落")
    
    if len(text_chunks) == 0:
        print("⚠️ 警告: 文本分割后没有有效内容")
        return
    
    audio_segments = []
    successful_chunks = 0
    
    for i, chunk in enumerate(text_chunks):
        if not chunk.strip():  # 跳过空段落
            continue
            
        print(f"\n生成第 {i+1}/{len(text_chunks)} 段音频...")
        print(f"段落内容预览: {chunk[:80]}..." if len(chunk) > 80 else f"段落内容: {chunk}")
        print(f"段落长度: {len(chunk)} 字符")
        
        # 生成音频
        audio_data = generate_audio(api_url, prompt_audio_path, prompt_text, chunk)
        
        if audio_data:
            # 保存临时音频文件
            temp_path = f"temp_{txt_path.stem}_{i}.wav"
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            # 加载音频段
            try:
                audio_segment = AudioSegment.from_wav(temp_path)
                audio_segments.append(audio_segment)
                successful_chunks += 1
                print(f"✓ 第 {i+1} 段音频生成成功，时长: {len(audio_segment)/1000:.2f}秒")
            except Exception as e:
                print(f"✗ 加载音频段失败: {e}")
            
            # 删除临时文件
            try:
                os.remove(temp_path)
            except:
                pass
        else:
            print(f"✗ 第 {i+1} 段音频生成失败")
    
    if audio_segments:
        # 合并所有音频段
        print(f"\n开始合并 {len(audio_segments)} 个音频段...")
        combined_audio = audio_segments[0]
        
        for i, segment in enumerate(audio_segments[1:], 1):
            combined_audio += segment
            print(f"已合并第 {i+1}/{len(audio_segments)} 段")
        
        # 保存最终文件
        output_path = output_dir / f"{txt_path.stem}.wav"
        combined_audio.export(output_path, format="wav")
        print(f"✓ 合并完成! 总时长: {len(combined_audio)/1000:.2f}秒")
        print(f"✓ 音频已保存到: {output_path}")
    else:
        print("✗ 未能生成任何音频段")

def main():
    parser = argparse.ArgumentParser(description='批量处理txt文件生成音频')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='包含txt文件的目录（如：No_6_Light_Novel_Xiang_IndexTTS2_Chapter1）')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='输出wav文件的目录')
    parser.add_argument('--api_url', type=str, default="http://localhost:8000", 
                       help='TTS API地址')
    parser.add_argument('--prompt_audio', type=str, default="类王翔音频_vocals.wav", 
                       help='提示音频文件路径')
    parser.add_argument('--prompt_text', type=str, 
                       default="因为刚刚失恋根本没有心情打理自己，一个是生活一个是感情。", 
                       help='提示文本')
    parser.add_argument('--lines_per_chunk', type=int, default=10,
                       help='每个音频段落的行数（默认：10）')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    # 检查提示音频文件是否存在
    if not Path(args.prompt_audio).exists():
        print(f"错误: 提示音频文件 {args.prompt_audio} 不存在")
        return
    
    # 查找所有txt文件
    txt_files = list(input_dir.glob('*.txt'))
    print(f"找到 {len(txt_files)} 个txt文件")
    
    if len(txt_files) == 0:
        print(f"在目录 {input_dir} 中未找到txt文件")
        return
    
    total_start_time = time.time()
    
    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n{'='*60}")
        print(f"处理进度: {i}/{len(txt_files)} - {txt_file.name}")
        print(f"{'='*60}")
        
        file_start_time = time.time()
        process_txt_file(txt_file, output_dir, args.api_url, 
                        args.prompt_audio, args.prompt_text, args.lines_per_chunk)
        file_time = time.time() - file_start_time
        print(f"文件处理时间: {file_time:.2f}秒")
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"所有文件处理完成!")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
