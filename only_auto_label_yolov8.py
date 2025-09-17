import argparse
from pathlib import Path
from ultralytics import YOLO
import shutil
import os

def auto_annotate(model_path, source_dir, output_dir, save_vis=True, save_conf=True, expected_columns=13):
    """
    使用训练好的YOLOv8模型对图像进行自动标注（推理），并保存有有效目标的原始图像。
    同时修复标签格式问题，确保每行有正确的字段数。
    
    参数:
        model_path (str): 训练好的模型权重文件路径
        source_dir (str): 包含未标注图像的源目录路径
        output_dir (str): 保存输出标签、可视化结果和原始图像的根目录
        save_vis (bool): 是否保存带预测结果的可视化图像，用于人工检查
        save_conf (bool): 是否在生成的标签文件中保存置信度
        expected_columns (int): 每行期望的列数（默认13）
    """
    
    # 转换为Path对象以便处理路径
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}。请提供正确的模型路径。")
    
    # 创建输出目录（如果不存在）
    labels_output_dir = output_path / "labels"  # 保存标签文件
    vis_output_dir = output_path / "vis"       # 保存可视化图像（可选）
    images_output_dir = output_path / "images" # 保存有有效目标的原始图像
    
    labels_output_dir.mkdir(parents=True, exist_ok=True)
    if save_vis:
        vis_output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"模型路径: {model_path}")
    print(f"源图像目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"期望的标签列数: {expected_columns}")
    
    try:
        # 加载训练好的模型
        model = YOLO(model_path)
        print("模型加载成功!")
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}。请检查模型路径和格式。")
    
    # 进行预测（推理）
    try:
        results = model.predict(
            source=source_dir,
            save=save_vis,          # 保存可视化图像
            save_txt=True,          # 将预测结果保存为.txt标签文件
            save_conf=save_conf,    # 在标签中保存置信度
            project=output_dir,     # 项目根目录
            name="predictions",     # 此次预测运行的名称
            exist_ok=True           # 允许覆盖现有目录
        )
        print("模型预测完成!")
    except Exception as e:
        raise RuntimeError(f"模型预测失败: {str(e)}")
    
    # 定义YOLOv8默认保存的路径
    default_prediction_dir = output_path / "predictions"
    default_labels_dir = default_prediction_dir / "labels"
    default_vis_dir = default_prediction_dir
    
    # 检查并修复标签格式
    if default_labels_dir.exists():
        fixed_count = 0
        for label_file in default_labels_dir.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    
                    # 修复标签格式：确保每行有expected_columns个字段
                    if len(parts) != expected_columns:
                        # 如果字段不足，补充0值
                        if len(parts) < expected_columns:
                            parts.extend(['0'] * (expected_columns - len(parts)))
                        # 如果字段过多，截断
                        else:
                            parts = parts[:expected_columns]
                        
                        fixed_count += 1
                        print(f"修复标签格式: {label_file.name} -> {len(parts)}列")
                    
                    new_lines.append(" ".join(parts) + "\n")
                
                # 写入修复后的标签
                with open(label_file, 'w') as f:
                    f.writelines(new_lines)
                    
            except Exception as e:
                print(f"处理标签文件 {label_file.name} 时出错: {str(e)}")
        
        if fixed_count > 0:
            print(f"已修复 {fixed_count} 个标签文件的格式问题")
    
    # 检查并复制有有效目标的原始图像到images文件夹
    if default_labels_dir.exists():
        copied_count = 0
        for label_file in default_labels_dir.glob("*.txt"):
            stem = label_file.stem  # 获取文件名（不带扩展名）
            
            # 在源目录中查找对应图像文件（支持常见图像格式）
            found = False
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]:
                potential_image_path = source_path / (stem + ext)
                if potential_image_path.exists():
                    # 复制原图到images文件夹
                    shutil.copy2(str(potential_image_path), str(images_output_dir / (stem + ext)))
                    copied_count += 1
                    found = True
                    break
            
            if not found:
                print(f"警告: 未找到与标签文件 {label_file.name} 对应的图像文件")
        
        print(f"已复制 {copied_count} 个有有效目标的原始图像")
    
    # 检查并移动标签文件
    if default_labels_dir.exists():
        moved_count = 0
        for label_file in default_labels_dir.glob("*.txt"):
            shutil.move(str(label_file), str(labels_output_dir / label_file.name))
            moved_count += 1
        
        # 删除空的默认labels目录
        try:
            default_labels_dir.rmdir()
        except OSError:
            pass  # 目录可能不为空，忽略错误
        
        print(f"已移动 {moved_count} 个标签文件")
    
    # 检查并移动可视化图像（如果要求保存）
    if save_vis and default_vis_dir.exists():
        moved_vis_count = 0
        for img_file in default_vis_dir.glob("*.*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                shutil.move(str(img_file), str(vis_output_dir / img_file.name))
                moved_vis_count += 1
        
        print(f"已移动 {moved_vis_count} 个可视化图像")
    
    # 删除空的默认预测目录
    if default_prediction_dir.exists():
        try:
            default_prediction_dir.rmdir()
        except OSError:
            print(f"注意: 目录 {default_prediction_dir} 不为空，未能删除")
    
    # 验证最终输出
    labels_count = len(list(labels_output_dir.glob("*.txt")))
    images_count = len(list(images_output_dir.glob("*.*")))
    vis_count = len(list(vis_output_dir.glob("*.*"))) if save_vis else 0
    
    print(f"\n[完成] 自动标注完成!")
    print(f"生成的标签文件: {labels_count} 个 (位于 {labels_output_dir})")
    print(f"有有效目标的原始图像: {images_count} 个 (位于 {images_output_dir})")
    if save_vis:
        print(f"可视化结果: {vis_count} 个 (位于 {vis_output_dir})")
    
    # 验证标签格式
    if labels_count > 0:
        print(f"\n标签格式验证:")
        sample_label = next(labels_output_dir.glob("*.txt"))
        with open(sample_label, 'r') as f:
            sample_line = f.readline().strip()
            columns = len(sample_line.split())
            print(f"样本标签: {sample_label.name}")
            print(f"每行列数: {columns} (期望: {expected_columns})")
            print(f"格式{'正确' if columns == expected_columns else '可能有问题'}")
    
    return {
        "labels_dir": str(labels_output_dir),
        "images_dir": str(images_output_dir),
        "vis_dir": str(vis_output_dir) if save_vis else None,
        "labels_count": labels_count,
        "images_count": images_count,
        "vis_count": vis_count
    }

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description='使用YOLOv8模型对图像进行自动标注（推理），修复标签格式问题，并保存有有效目标的原始图像。'
    )
    # parser.add_argument('--model', type=str, default='E:/ZichenFeng/RobotMaster/runs/pose/train/weights/best.pt')
    parser.add_argument('--model', type=str, default='./zichen/models/yolov8_448x448_buff_merge_GRAY_center/train/weights/last.pt')

    parser.add_argument('--source', type=str, default='./zichen/dataset/test_buff_energy/images')
    parser.add_argument('--output', type=str, default='./zichen/auto_label_outputs/auto_annotate_output',
                       help='保存输出结果的根目录，默认: ./zichen/auto_label_outputs/auto_annotate_output')
    parser.add_argument('--no-vis', action='store_false', dest='save_vis',
                       help='不保存带预测结果的可视化图像')
    parser.add_argument('--no-conf', action='store_false', dest='save_conf',
                       help='不在生成的标签文件中保存置信度')
    parser.add_argument('--columns', type=int, default=13,
                       help='每行期望的列数（默认: 13）')
    
    args = parser.parse_args()
    
    # 运行自动标注函数
    try:
        result = auto_annotate(
            model_path=args.model,
            source_dir=args.source,
            output_dir=args.output,
            save_vis=args.save_vis,
            save_conf=args.save_conf,
            expected_columns=args.columns
        )
        
        print(f"\n下一步: 请检查 {result['labels_dir']} 中的标签文件，确保格式正确。")
        print("然后可以使用这些数据训练最终模型。")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        print("请检查: 1) 模型路径是否正确 2) 源图像目录是否存在 3) 模型格式是否兼容")
