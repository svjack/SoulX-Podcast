'''
python epub_batch_converter.py --batch --source-dir Light_Novel_2025_11_epub --target-dir Light_Novel_2025_11_txt --lines 0
'''

import os
import re
from pathlib import Path
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup
import argparse

def convert_epub_to_txt(epub_path, output_path, lines_per_file=64, remove_adjacent_duplicates=True):
    """
    将EPUB文件转换为TXT文件，支持相邻行去重和按行数分割[1,4](@ref)
    """
    try:
        # 读取EPUB文件[1,4](@ref)
        book = epub.read_epub(epub_path)
        all_lines = []

        # 遍历EPUB中的所有项目[1](@ref)
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # 使用BeautifulSoup解析HTML内容，提取纯文本[1,4](@ref)
                soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                text = soup.get_text()

                # 按行分割，并清理每一行的空白字符
                lines = text.splitlines()
                cleaned_lines = [line.strip() for line in lines if line.strip()]
                all_lines.extend(cleaned_lines)

        if not all_lines:
            print("警告: 未从EPUB文件中提取到任何文本内容。")
            return False

        # 处理相邻重复行
        if remove_adjacent_duplicates:
            unique_lines = []
            previous_line = None
            for line in all_lines:
                if line != previous_line:
                    unique_lines.append(line)
                    previous_line = line
            all_lines = unique_lines
            print(f"相邻去重后，总行数: {len(all_lines)}")

        # 确定输出模式并保存
        output_dir = Path(output_path)
        if lines_per_file > 0 and len(all_lines) > lines_per_file:
            output_dir.mkdir(parents=True, exist_ok=True)
            return _save_split_files(all_lines, output_dir, lines_per_file)
        else:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            return _save_single_file(all_lines, output_dir)

    except Exception as e:
        print(f"处理EPUB文件时出错: {e}")
        return False

def _save_single_file(lines, output_path):
    """将所有行保存到单个TXT文件中[1](@ref)"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:  # 确保使用UTF-8编码[5](@ref)
            for line in lines:
                f.write(line + '\n')
        print(f"✓ 已生成单个TXT文件: {output_path}")
        print(f"✓ 文件总行数: {len(lines)}")
        return True
    except Exception as e:
        print(f"保存单个文件时出错: {e}")
        return False

def _save_split_files(lines, output_dir, lines_per_file):
    """将行列表按指定行数分割，并保存为多个编号的TXT文件"""
    try:
        total_files = (len(lines) + lines_per_file - 1) // lines_per_file
        files_created = 0

        for i in range(0, len(lines), lines_per_file):
            chunk = lines[i:i + lines_per_file]
            file_number = str(files_created).zfill(6)
            filename = f"{file_number}.txt"
            filepath = output_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                for line in chunk:
                    f.write(line + '\n')

            files_created += 1
            print(f"✓ 生成分割文件: {filepath} (包含 {len(chunk)} 行)")

        print(f"✓ 分割完成! 共生成 {files_created} 个文件到目录 {output_dir}")
        print(f"✓ 所有文件总行数: {len(lines)}")
        return True

    except Exception as e:
        print(f"分割文件时出错: {e}")
        return False

def batch_convert_epub_folder(source_folder, target_folder, lines_per_file=64, remove_duplicates=True):
    """
    批量转换文件夹中的所有EPUB文件为TXT格式[1](@ref)

    Args:
        source_folder (str): 包含EPUB文件的源文件夹路径
        target_folder (str): 保存TXT文件的目标文件夹路径
        lines_per_file (int): 每个分割文件的行数
        remove_duplicates (bool): 是否移除相邻重复行
    """
    source_path = Path(source_folder)
    target_path = Path(target_folder)

    # 确保源目录存在
    if not source_path.exists():
        print(f"❌ 源目录不存在: {source_folder}")
        return False

    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)

    # 查找所有EPUB文件
    epub_files = list(source_path.glob("**/*.epub"))

    if not epub_files:
        print(f"❌ 在目录 {source_folder} 中未找到任何EPUB文件")
        return False

    print(f"📁 找到 {len(epub_files)} 个EPUB文件")
    success_count = 0

    for epub_file in epub_files:
        print(f"\n🔄 正在处理: {epub_file.name}")

        # 生成输出路径（保持目录结构）
        relative_path = epub_file.relative_to(source_path)
        output_name = relative_path.with_suffix('')

        if lines_per_file > 0:
            # 分割模式：创建子目录
            output_dir = target_path / output_name
            output_path = output_dir
        else:
            # 单个文件模式
            output_path = target_path / output_name.with_suffix('.txt')

        # 执行转换
        if convert_epub_to_txt(
            epub_path=str(epub_file),
            output_path=str(output_path),
            lines_per_file=lines_per_file,
            remove_adjacent_duplicates=remove_duplicates
        ):
            success_count += 1

    print(f"\n🎉 批量转换完成！")
    print(f"📊 成功转换: {success_count}/{len(epub_files)} 个文件")
    print(f"📁 输出目录: {target_folder}")

    return success_count > 0

def main():
    """主函数，支持命令行参数和批量处理"""
    parser = argparse.ArgumentParser(description='将EPUB文件转换为TXT文件，支持去重、分割和批量处理')
    parser.add_argument('--input', help='输入的EPUB文件路径或包含EPUB文件的目录路径')
    parser.add_argument('--output', help='输出的TXT文件路径（单文件）或存放转换结果的目录路径（批量）')
    parser.add_argument('--source-dir', default='Light_Novel_2025_11_epub',
                       help='包含EPUB文件的源目录，默认: Light_Novel_2025_11_epub')
    parser.add_argument('--target-dir', default='Light_Novel_2025_11_txt',
                       help='保存TXT文件的目标目录，默认: Light_Novel_2025_11_txt')
    parser.add_argument('--lines', type=int, default=64,
                       help='每个分割文件的行数。设置为0则输出单个TXT文件。默认: 64')
    parser.add_argument('--keep-duplicates', action='store_true',
                       help='使用此选项将保留相邻的重复行，默认行为是去除相邻重复行')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式，转换整个目录中的EPUB文件')

    args = parser.parse_args()

    if args.batch:
        # 批量处理模式
        success = batch_convert_epub_folder(
            source_folder=args.source_dir,
            target_folder=args.target_dir,
            lines_per_file=args.lines,
            remove_duplicates=not args.keep_duplicates
        )
    elif args.input and args.output:
        # 单文件处理模式
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"错误: 输入文件不存在: {args.input}")
            return

        success = convert_epub_to_txt(
            epub_path=args.input,
            output_path=args.output,
            lines_per_file=args.lines,
            remove_adjacent_duplicates=not args.keep_duplicates
        )
    else:
        print("请指定处理模式：")
        print("  单文件: --input <epub文件> --output <输出路径>")
        print("  批量处理: --batch [--source-dir 输入目录] [--target-dir 输出目录]")
        return

    if success:
        print("✨ 处理成功完成！")
    else:
        print("❌ 处理过程中出现问题。")

if __name__ == "__main__":
    main()
