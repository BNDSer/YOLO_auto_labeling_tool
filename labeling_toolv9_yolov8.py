import sys
import os
import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QListWidget, QFileDialog, QMessageBox,
                             QInputDialog, QSpinBox, QTreeWidget, QTreeWidgetItem, QSplitter,
                             QProgressBar, QStatusBar, QToolBar, QAction, QDockWidget, QComboBox)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QIcon, QCursor
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
import cv2
import threading
import shutil
import tempfile
from pathlib import Path
from PyQt5.QtCore import QTimer
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

class KeypointAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.keypoints_list = QListWidget(self)
        self.setWindowTitle("YOLOv8 Keypoints 标注工具 - 增强版")
        self.setGeometry(100, 100, 1400, 800)
        
        # 初始化变量
        self.image_dir = ""
        self.labels_dir = ""
        self.image_files = []
        self.current_image_index = -1
        self.current_image = None
        self.scale_factor = 1.0
        self.original_image = None

        # AI model path
        self.model_path = ""  # 用户选择的 .pt 模型路径
        
        # 标注数据
        self.annotations = []
        self.current_annotation = None
        self.selected_point_index = -1
        self.dragging = False
        
        # 类别和关键点配置（启动时为空，导入标签后自动扩展）
        self.categories = []
        self.current_category_id = 0
        
        self.init_ui()
        
    def init_ui(self):
        # 创建中央窗口和主布局
        self.keypoints_list = QListWidget(self)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧面板 - 文件浏览和类别管理
        left_dock = QDockWidget("文件浏览", self)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 文件夹选择按钮
        self.btn_open_folder = QPushButton("打开图像文件夹")
        self.btn_open_folder.clicked.connect(self.open_image_folder)
        left_layout.addWidget(self.btn_open_folder)
        
        # 标签文件夹选择按钮
        self.btn_open_labels_folder = QPushButton("选择标签文件夹")
        self.btn_open_labels_folder.clicked.connect(self.open_labels_folder)
        left_layout.addWidget(self.btn_open_labels_folder)
        
        # 当前目录显示
        self.lbl_current_labels_dir = QLabel("未选择标签目录")
        left_layout.addWidget(self.lbl_current_labels_dir)

        # ===== 新增：模型选择与 AI 自动标注按钮 =====
        self.btn_select_model = QPushButton("选择 .pt 模型")
        self.btn_select_model.clicked.connect(self.select_model)
        left_layout.addWidget(self.btn_select_model)

        self.lbl_model_path = QLabel("未选择模型")
        left_layout.addWidget(self.lbl_model_path)

        self.btn_auto_annotate = QPushButton("AI 标注全部图像")
        self.btn_auto_annotate.clicked.connect(self.auto_annotate_all)
        left_layout.addWidget(self.btn_auto_annotate)
        # ============================================
        
        # 文件列表
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self.load_image)
        left_layout.addWidget(QLabel("图像文件:"))
        left_layout.addWidget(self.file_list)
        
        # 标签操作按钮
        self.btn_load_labels = QPushButton("导入标签文件")
        self.btn_load_labels.clicked.connect(self.load_labels_for_current_image)
        left_layout.addWidget(self.btn_load_labels)
        
        self.btn_save_labels = QPushButton("保存标签文件")
        self.btn_save_labels.clicked.connect(self.save_annotations)
        left_layout.addWidget(self.btn_save_labels)
        
        # 类别选择
        left_layout.addWidget(QLabel("物体类别:"))
        self.category_combo = QComboBox()
        self.update_category_combo()
        self.category_combo.currentIndexChanged.connect(self.category_changed)
        left_layout.addWidget(self.category_combo)
        
        # 关键点列表
        left_layout.addWidget(QLabel("关键点列表:"))
        self.keypoints_list = QListWidget()
        left_layout.addWidget(self.keypoints_list)
        
        # 标注目标列表（支持多目标切换）
        left_layout.addWidget(QLabel("标注目标列表:"))
        self.annotation_list = QListWidget()
        self.annotation_list.currentRowChanged.connect(self.switch_annotation)
        left_layout.addWidget(self.annotation_list)
        
        left_dock.setWidget(left_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)
        
        # 中央图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.image_label.mousePressEvent = self.image_mouse_press
        self.image_label.mouseMoveEvent = self.image_mouse_move
        self.image_label.mouseReleaseEvent = self.image_mouse_release
        main_layout.addWidget(self.image_label)
        
        # 右侧面板 - 标注操作
        right_dock = QDockWidget("标注操作", self)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 标注控制按钮
        self.btn_new_annotation = QPushButton("新建标注")
        self.btn_new_annotation.clicked.connect(self.start_new_annotation)
        right_layout.addWidget(self.btn_new_annotation)
        
        self.btn_save_annotations = QPushButton("保存标注")
        self.btn_save_annotations.clicked.connect(self.save_annotations)
        right_layout.addWidget(self.btn_save_annotations)
        
        self.btn_undo = QPushButton("撤销上一点")
        self.btn_undo.clicked.connect(self.undo_last_point)
        right_layout.addWidget(self.btn_undo)
        
        self.btn_clear = QPushButton("清除当前标注")
        self.btn_clear.clicked.connect(self.clear_current_annotation)
        right_layout.addWidget(self.btn_clear)
        
        # 类别管理按钮
        right_layout.addWidget(QLabel("类别管理:"))
        self.btn_add_category = QPushButton("添加新类别")
        self.btn_add_category.clicked.connect(self.add_new_category)
        right_layout.addWidget(self.btn_add_category)
        
        self.btn_edit_category = QPushButton("编辑当前类别")
        self.btn_edit_category.clicked.connect(self.edit_current_category)
        right_layout.addWidget(self.btn_edit_category)
        
        right_dock.setWidget(right_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
        # 工具栏
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # 添加快捷键说明
        self.status_bar.showMessage("快捷键: S-保存 | N-新建标注 | U-撤销 | C-清除 | ←/→-切换图像")
    
    def open_labels_folder(self):
        """手动选择标签文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择标签文件夹")
        if folder_path:
            self.labels_dir = folder_path
            self.lbl_current_labels_dir.setText(f"标签目录: {os.path.basename(folder_path)}")
            self.status_bar.showMessage(f"已选择标签目录: {folder_path}")
    
    def get_labels_dir(self):
        """获取标签目录路径（优先使用手动选择的目录）"""
        if self.labels_dir:
            return self.labels_dir
        elif self.image_dir:
            # 如果没有手动选择标签目录，尝试自动创建与images同级的labels目录
            if "images" in self.image_dir:
                labels_dir = self.image_dir.replace("images", "labels")
            else:
                labels_dir = os.path.join(os.path.dirname(self.image_dir), "labels")
            
            # 确保labels目录存在
            if not os.path.exists(labels_dir):
                os.makedirs(labels_dir)
                self.status_bar.showMessage(f"已自动创建labels目录: {labels_dir}")
            
            return labels_dir
        else:
            return ""
    
    def get_label_path(self, image_name):
        """根据图像文件名获取对应的标签文件路径"""
        if not image_name:
            return ""
        
        base_name = os.path.splitext(image_name)[0]
        labels_dir = self.get_labels_dir()
        if not labels_dir:
            return ""
        
        return os.path.join(labels_dir, f"{base_name}.txt")
    
    def update_category_combo(self):
        self.category_combo.clear()
        for category in self.categories:
            self.category_combo.addItem(category["name"])
        self.update_keypoints_list()
    
    def category_changed(self, index):
        """切换类别时：更新当前类别、关键点列表，若存在当前标注则修改其类别并同步关键点个数"""
        self.current_category_id = index
        self.update_keypoints_list()

        # 如果当前有选中的标注，修改该标注的 category_id 并调整 keypoints 数量
        if self.current_annotation is not None and 0 <= index < len(self.categories):
            self.current_annotation["category_id"] = index
            needed = len(self.categories[index]["keypoints"])
            # 补齐或截断关键点列表
            kps = self.current_annotation.get("keypoints", [])
            while len(kps) < needed:
                kps.append([0.0, 0.0, 0])
            self.current_annotation["keypoints"] = kps[:needed]
            self.update_display()
            self.status_bar.showMessage(f"当前标注类别已设为: {self.categories[index]['name']}")
    
    def update_keypoints_list(self):
        self.keypoints_list.clear()
        if 0 <= self.current_category_id < len(self.categories):
            current_category = self.categories[self.current_category_id]
            for i, kp_name in enumerate(current_category["keypoints"]):
                self.keypoints_list.addItem(f"{i}: {kp_name}")
    
    def open_image_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder_path:
            self.image_dir = folder_path
            self.image_files = [f for f in os.listdir(folder_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            self.file_list.clear()
            self.file_list.addItems(self.image_files)
            if self.image_files:
                self.file_list.setCurrentRow(0)
    
    def load_image(self, index):
        if 0 <= index < len(self.image_files):
            self.current_image_index = index
            image_path = os.path.join(self.image_dir, self.image_files[index])
            
            # 使用OpenCV读取图像
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                QMessageBox.warning(self, "错误", f"无法加载图像: {image_path}")
                return
            
            # 转换颜色空间（OpenCV是BGR，需要转RGB）
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            
            # 创建QImage并显示
            qt_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # 缩放图像以适应标签大小
            self.current_image = pixmap
            self.display_image()
            
            # 自动加载对应的标注文件（如果存在）
            self.load_annotation_file()
    
    def display_image(self):
        if self.current_image:
            # 缩放图像以适应显示区域
            scaled_pixmap = self.current_image.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.scale_factor = scaled_pixmap.width() / self.current_image.width()
            self.update_display()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.display_image()
    
    def image_mouse_press(self, event):
        if not self.current_image or not self.current_annotation:
            return

        pos = event.pos()
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return

        img_x = (self.image_label.width() - pixmap.width()) / 2
        img_y = (self.image_label.height() - pixmap.height()) / 2

        x_img = (pos.x() - img_x) / self.scale_factor
        y_img = (pos.y() - img_y) / self.scale_factor

        # 判断是否点中了某个关键点（允许拖拽）
        found = False
        for i, (x, y, v) in enumerate(self.current_annotation["keypoints"]):
            if v > 0:
                kp_x = x * self.current_image.width()
                kp_y = y * self.current_image.height()
                dist = ((kp_x - x_img) ** 2 + (kp_y - y_img) ** 2) ** 0.5
                if dist < 10:  # 允许10像素范围内拖拽
                    self.selected_point_index = i
                    self.dragging = True
                    found = True
                    break

        if not found and event.button() == Qt.LeftButton:
            # 没点中任何关键点，则添加新点（原逻辑）
            if self.current_annotation and 0 <= self.current_category_id < len(self.categories):
                current_category = self.categories[self.current_category_id]
                for i in range(len(current_category["keypoints"])):
                    if i >= len(self.current_annotation["keypoints"]) or self.current_annotation["keypoints"][i][2] == 0:
                        if i >= len(self.current_annotation["keypoints"]):
                            while len(self.current_annotation["keypoints"]) <= i:
                                self.current_annotation["keypoints"].append([0, 0, 0])
                        self.current_annotation["keypoints"][i] = [
                            x_img / self.current_image.width(),
                            y_img / self.current_image.height(),
                            2
                        ]
                        self.selected_point_index = i
                        self.update_display()
                        break

    def image_mouse_move(self, event):
        if self.dragging and self.selected_point_index >= 0 and self.current_annotation:
            pos = event.pos()
            pixmap = self.image_label.pixmap()
            if not pixmap:
                return
            img_x = (self.image_label.width() - pixmap.width()) / 2
            img_y = (self.image_label.height() - pixmap.height()) / 2
            x_img = (pos.x() - img_x) / self.scale_factor
            y_img = (pos.y() - img_y) / self.scale_factor
            # 更新关键点坐标
            self.current_annotation["keypoints"][self.selected_point_index][0] = x_img / self.current_image.width()
            self.current_annotation["keypoints"][self.selected_point_index][1] = y_img / self.current_image.height()
            self.update_display()

    def image_mouse_release(self, event):
        self.dragging = False
        self.selected_point_index = -1
    
    def start_new_annotation(self):
        current_category_id = self.category_combo.currentIndex()
        if current_category_id < 0:
            QMessageBox.warning(self, "警告", "请先选择一个物体类别")
            return
            
        # 创建新极标注
        self.current_annotation = {
            "category_id": current_category_id,
            "keypoints": []  # 格式: [[x1, y1, v1], [x2, y2, v2], ...]
        }
        
        # 初始化关键点列表（全部不可见）
        if 0 <= current_category_id < len(self.categories):
            current_category = self.categories[current_category_id]
            for _ in current_category["keypoints"]:
                self.current_annotation["keypoints"].append([0, 0, 0])  # x, y, visibility
        
        self.annotations.append(self.current_annotation)
        self.update_display()
        self.status_bar.showMessage("新建标注已创建，请点击图像添加关键点")
        self.refresh_annotation_list()
    
    def undo_last_point(self):
        if self.current_annotation:
            # 找到最后一个可见的点并将其设置为不可见
            for i in range(len(self.current_annotation["keypoints"]) - 1, -1, -1):
                if self.current_annotation["keypoints"][i][2] > 0:
                    self.current_annotation["keypoints"][i][2] = 0  # 设置为不可见
                    self.update_display()
                    self.status_bar.showMessage(f"已撤销关键点 {i}")
                    return
            
            self.status_bar.showMessage("没有可撤销的关键点")
    
    def clear_current_annotation(self):
        if self.current_annotation and self.annotations:
            self.annotations.remove(self.current_annotation)
            self.current_annotation = None
            if self.annotations:
                self.current_annotation = self.annotations[-1]
            self.update_display()
            self.status_bar.showMessage("当前标注已清除")
            self.refresh_annotation_list()
    
    def update_display(self):
        if not self.current_image:
            return
        
        display_pixmap = self.current_image.copy()
        painter = QPainter(display_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        for annotation in self.annotations:
            category_id = annotation["category_id"]
            keypoints = annotation["keypoints"]
            colors = [QColor(0, 255, 0), QColor(255, 0, 0), QColor(0, 0, 255), 
                     QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255)]
            color = colors[category_id % len(colors)]
            pen = QPen(color, 2)
            painter.setPen(pen)
            for i, kp in enumerate(keypoints):
                try:
                    x = float(kp[0])
                    y = float(kp[1])
                    visible = int(kp[2]) if len(kp) > 2 else 2
                except Exception:
                    continue
                if visible > 0:
                    x_pix = x * display_pixmap.width()
                    y_pix = y * display_pixmap.height()
                    painter.drawEllipse(QPoint(int(x_pix), int(y_pix)), 5, 5)
                    painter.setFont(QFont("Arial", 10))
                    painter.drawText(QPoint(int(x_pix) + 8, int(y_pix) - 8), str(i))
        painter.end()
        scaled_pixmap = display_pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    
    def load_labels_for_current_image(self):
        """导入当前图像的标签文件（兼容读取并恢复原有类别序号）"""
        if not self.image_files or self.current_image_index < 0:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return False
            
        image_name = self.image_files[self.current_image_index]
        txt_path = self.get_label_path(image_name)
        
        if not txt_path:
            QMessageBox.warning(self, "警告", "请先选择标签目录")
            return False
        
        if not os.path.exists(txt_path):
            reply = QMessageBox.question(self, "文件不存在", 
                                        f"标签文件 {txt_path} 不存在。是否创建新文件？",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return False
            # 创建空文件
            open(txt_path, 'w').close()
        
        try:
            self.annotations = []
            self.current_annotation = None
            
            # 获取当前图像的宽高（像素），用于判断/归一化
            if self.original_image is not None:
                img_h, img_w = self.original_image.shape[:2]
            else:
                img_w = self.current_image.width() if self.current_image else 1
                img_h = self.current_image.height() if self.current_image else 1
            
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:  # 至少需要类别ID和bbox
                        continue
                        
                    # 解析YOLOv8格式
                    category_id = int(parts[0])
                    bbox = list(map(float, parts[1:5]))
                    keypoints = []
                    
                    # 解析关键点（如果有），兼容两种格式： (x y v) 或 (x y)
                    if len(parts) > 5:
                        keypoints = self.parse_keypoints_with_v(parts[5:], img_w, img_h)
                    
                    # 如果读入的是像素坐标（存在>1的坐标），则转换为归一化坐标
                    normalized_keypoints = []
                    for x, y, v in keypoints:
                        if x > 1.0 or y > 1.0:
                            nx = x / img_w
                            ny = y / img_h
                        else:
                            nx = x
                            ny = y
                        normalized_keypoints.append([nx, ny, v])
                    
                    # 如果文件中包含比现有 categories 更多的类别索引，则自动创建占位类别以保留原类别序号
                    # 在解析完 normalized_keypoints 后
                    if category_id >= len(self.categories):
                        for cid in range(len(self.categories), category_id + 1):
                            kp_count = max(1, len(normalized_keypoints))
                            placeholder_kps = [f"kp{i}" for i in range(kp_count)]
                            self.categories.append({
                                "name": f"class_{cid}",
                                "keypoints": placeholder_kps
                            })
                        self.update_category_combo()
                    else:
                        needed = len(self.categories[category_id]["keypoints"])
                        if len(normalized_keypoints) > needed:
                            extra = len(normalized_keypoints) - needed
                            for _ in range(extra):
                                self.categories[category_id]["keypoints"].append(f"kp{needed}")
                                needed += 1
                            self.update_category_combo()

                    # 补齐关键点数以匹配类别定义
                    if 0 <= category_id < len(self.categories):
                        needed = len(self.categories[category_id]["keypoints"])
                        while len(normalized_keypoints) < needed:
                            normalized_keypoints.append([0.0, 0.0, 0])
                        # 不截断，保留所有点
                        # normalized_keypoints = normalized_keypoints[:needed]
                    
                    annotation = {
                        "category_id": category_id,
                        "bbox": bbox,
                        "keypoints": normalized_keypoints
                    }
                    self.annotations.append(annotation)
            
            if self.annotations:
                # 默认选中第一个标注
                self.current_annotation = self.annotations[0]
                self.current_category_id = self.current_annotation["category_id"]
                self.category_combo.setCurrentIndex(self.current_category_id)
                # 刷新标注列表
                self.refresh_annotation_list()
                
                self.update_display()
                self.status_bar.showMessage(f"已自动加载标注: {os.path.basename(txt_path)}")
                self.refresh_annotation_list()
                return True
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载标注文件时出错: {str(e)}")
            return False
    
    def save_annotations(self):
        """保存标注到指定的标签文件（统一 6 位小数，输出 class x_center y_center w h kp1_x kp1_y ...）"""
        if not self.image_files or self.current_image_index < 0:
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return False

        image_name = self.image_files[self.current_image_index]
        txt_path = self.get_label_path(image_name)

        if not txt_path:
            QMessageBox.warning(self, "警告", "请先选择标签目录")
            return False

        try:
            # 获取图像像素大小，用于将像素坐标归一化（如果检测到有像素坐标）
            if self.original_image is not None:
                img_h, img_w = self.original_image.shape[:2]
            else:
                img_w = self.current_image.width() if self.current_image else 1
                img_h = self.current_image.height() if self.current_image else 1

            with open(txt_path, 'w') as f:
                for annotation in self.annotations:
                    category_id = int(annotation.get("category_id", 0))
                    keypoints = annotation.get("keypoints", [])

                    # 确保关键点数量等于类别定义数（不足补 0）
                    if 0 <= category_id < len(self.categories):
                        needed = len(self.categories[category_id]["keypoints"])
                    else:
                        needed = len(keypoints)
                    # 复制一份避免修改原数据结构顺序/长度
                    kps = [list(k) for k in keypoints]
                    while len(kps) < needed:
                        kps.append([0.0, 0.0, 0])
                    kps = kps[:needed]

                    # 计算边界框（基于可见关键点）
                    valid_points = [kp for kp in kps if kp[2] > 0]
                    if not valid_points:
                        # 没有可见关键点则跳过该 annotation
                        continue

                    # 提取并归一化坐标（如果检测到为像素坐标）
                    xs = []
                    ys = []
                    for kp in valid_points:
                        x, y = float(kp[0]), float(kp[1])
                        if x > 1.0 or y > 1.0:
                            nx = x / img_w
                            ny = y / img_h
                        else:
                            nx = x
                            ny = y
                        xs.append(nx)
                        ys.append(ny)

                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    x_center = (x_min + x_max) / 2.0
                    y_center = (y_min + y_max) / 2.0
                    width = (x_max - x_min)
                    height = (y_max - y_min)

                    # 写入类别ID和边界框（统一 6 位小数）
                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                    # 按类别关键点顺序写入关键点坐标：只写 x y（不可见点写 0 0），确保归一化并使用 6 位小数
                    for kp in kps:
                        x, y, v = float(kp[0]), float(kp[1]), int(kp[2])
                        if x > 1.0 or y > 1.0:
                            nx = x / img_w
                            ny = y / img_h
                        else:
                            nx = x
                            ny = y
                        if v > 0:
                            f.write(f" {nx:.6f} {ny:.6f}")
                        else:
                            f.write(f" {0.0:.6f} {0.0:.6f}")

                    f.write("\n")

            self.status_bar.showMessage(f"标注已保存: {txt_path}")
            return True

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存标注文件时出错: {str(e)}")
            return False
    
    def load_annotation_file(self):
        """自动加载标注文件（如果存在），保留原类别 id 并自动创建占位类别以兼容"""
        if not self.image_files or self.current_image_index < 0:
            return
            
        self.annotations = []
        self.current_annotation = None
        
        # 当前图像像素尺寸
        if self.original_image is not None:
            img_h, img_w = self.original_image.shape[:2]
        else:
            img_w = self.current_image.width() if self.current_image else 1
            img_h = self.current_image.height() if self.current_image else 1 
        
        image_name = self.image_files[self.current_image_index]
        txt_path = self.get_label_path(image_name)
        
        if txt_path and os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:  # 至少需要类别ID和bbox
                            continue
                            
                        category_id = int(parts[0])
                        bbox = list(map(float, parts[1:5]))
                        keypoints = []
                        
                        # 解析关键点（如果有），兼容 (x y v) 和 (x y)
                        if len(parts) > 5:
                            keypoints = self.parse_keypoints_with_v(parts[5:], img_w, img_h)
                        
                        # 将像素坐标转换为归一化（如果检测到 >1）
                        normalized_keypoints = []
                        for x, y, v in keypoints:
                            if x > 1.0 or y > 1.0:
                                nx = x / img_w
                                ny = y / img_h
                            else:
                                nx = x
                                ny = y
                            normalized_keypoints.append([nx, ny, v])
                        
                        # 如果文件中包含比现有 categories 更多的类别索引，则自动创建占位类别
                        # 在解析完 normalized_keypoints 后
                        if category_id >= len(self.categories):
                            for cid in range(len(self.categories), category_id + 1):
                                kp_count = max(1, len(normalized_keypoints))
                                placeholder_kps = [f"kp{i}" for i in range(kp_count)]
                                self.categories.append({
                                    "name": f"class_{cid}",
                                    "keypoints": placeholder_kps
                                })
                            self.update_category_combo()
                        else:
                            needed = len(self.categories[category_id]["keypoints"])
                            if len(normalized_keypoints) > needed:
                                extra = len(normalized_keypoints) - needed
                                for _ in range(extra):
                                    self.categories[category_id]["keypoints"].append(f"kp{needed}")
                                    needed += 1
                                self.update_category_combo()
                                                
                        # 补齐关键点数量
                        if 0 <= category_id < len(self.categories):
                            needed = len(self.categories[category_id]["keypoints"])
                            while len(normalized_keypoints) < needed:
                                normalized_keypoints.append([0.0, 0.0, 0])
                            # 不截断，保留所有点
                            # normalized_keypoints = normalized_keypoints[:needed]
                        
                        annotation = {
                            "category_id": category_id,
                            "bbox": bbox,
                            "keypoints": normalized_keypoints
                        }
                        self.annotations.append(annotation)
                
                if self.annotations:
                    self.current_annotation = self.annotations[-1]
                    self.current_category_id = self.current_annotation["category_id"]
                    self.category_combo.setCurrentIndex(self.current_category_id)
                
                self.update_display()
                self.status_bar.showMessage(f"已自动加载标注: {os.path.basename(txt_path)}")
                self.refresh_annotation_list()
                
            except Exception as e:
                self.status_bar.showMessage(f"加载标注文件时出错: {str(e)}")
    
    def add_new_category(self):
        name, ok = QInputDialog.getText(self, "添加新类别", "请输入类别名称:")
        if ok and name:
            keypoints, ok = QInputDialog.getText(self, "关键点设置", 
                                                "请输入关键点名称（用逗号分隔）:")
            if ok:
                kp_list = [kp.strip() for kp in keypoints.split(",") if kp.strip()]
                self.categories.append({
                    "name": name,
                    "keypoints": kp_list
                })
                self.update_category_combo()
                self.category_combo.setCurrentIndex(len(self.categories) - 1)
    
    def edit_current_category(self):
        current_index = self.category_combo.currentIndex()
        if 0 <= current_index < len(self.categories):
            category = self.categories[current_index]
            new_name, ok = QInputDialog.getText(self, "编辑类别", "类别名称:", 
                                               text=category["name"])
            if ok:
                current_kps = ",".join(category["keypoints"])
                new_kps, ok = QInputDialog.getText(self, "编辑关键点", "关键点名称（逗号分隔）:", 
                                                  text=current_kps)
                if ok:
                    kp_list = [kp.strip() for kp in new_kps.split(",") if kp.strip()]
                    category["name"] = new_name
                    category["keypoints"] = kp_list
                    self.update_category_combo()
    
    def select_model(self):
        """选择已训练好的 .pt 模型（独立于手动标注）"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件(.pt/.pth)", filter="模型文件 (*.pt *.pth);;所有文件 (*)")
        if file_path:
            self.model_path = file_path
            self.lbl_model_path.setText(os.path.basename(file_path))
            self.status_bar.showMessage(f"已选择模型: {file_path}")
            if not ULTRALYTICS_AVAILABLE:
                QMessageBox.warning(self, "依赖缺失", "ultralytics 库未检测到，自动标注功能将无法运行。请 pip install ultralytics")
    
    def auto_annotate_all(self):
        """开始对当前 image_dir 中所有图片进行 AI 标注（后台线程执行）"""
        if not ULTRALYTICS_AVAILABLE:
            QMessageBox.warning(self, "错误", "ultralytics 库未安装，无法执行自动标注。")
            return
        if not self.model_path:
            QMessageBox.warning(self, "错误", "请先选择 .pt 模型")
            return
        if not self.image_dir:
            QMessageBox.warning(self, "错误", "请先选择图像文件夹")
            return

        reply = QMessageBox.question(self, "确认", "开始对所有图像执行 AI 标注？这将生成/覆盖 labels 目录中的标签文件。继续？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        self.status_bar.showMessage("AI 标注进行中...（后台）")
        self.btn_auto_annotate.setEnabled(False)
        self.btn_select_model.setEnabled(False)
        thread = threading.Thread(target=self._auto_annotate_worker, daemon=True)
        thread.start()

    def _auto_annotate_worker(self):
        """后台 worker：调用 YOLO.predict 并移动生成的 labels 到 labels_dir"""
        success = False
        message = ""
        try:
            model = YOLO(self.model_path)
            # 使用临时目录保存 ultralytics 的预测输出
            tmp_root = tempfile.mkdtemp(prefix="auto_annot_")
            # 批量预测整个文件夹
            model.predict(
                source=self.image_dir,
                save=False,        # 不保存可视化到默认位置（可按需改）
                save_txt=True,     # 保存 txt 标签
                save_conf=False,   # 不保存置信度（与手动标注格式一致）
                project=tmp_root,
                name="predictions",
                exist_ok=True
            )
            default_labels_dir = Path(tmp_root) / "predictions" / "labels"
            target_labels_dir = Path(self.get_labels_dir())
            target_labels_dir.mkdir(parents=True, exist_ok=True)

            moved = 0
            if default_labels_dir.exists():
                for label_file in default_labels_dir.glob("*.txt"):
                    shutil.move(str(label_file), str(target_labels_dir / label_file.name))
                    moved += 1

            # 尝试清理临时预测目录
            try:
                shutil.rmtree(Path(tmp_root) / "predictions")
            except Exception:
                pass

            success = True
            message = f"AI 标注完成，已生成 {moved} 个标签文件，存放于: {target_labels_dir}"
        except Exception as e:
            message = f"AI 标注失败: {str(e)}"

        # 在主线程回调更新 UI / 弹窗
        QTimer.singleShot(0, lambda: self._on_auto_done(success, message))

    def _on_auto_done(self, success, message):
        self.btn_auto_annotate.setEnabled(True)
        self.btn_select_model.setEnabled(True)
        self.status_bar.showMessage(message)
        if success:
            # 修正所有标签文件为严格六位小数
            labels_dir = Path(self.get_labels_dir())
            for txt_file in labels_dir.glob("*.txt"):
                lines = []
                with open(txt_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        new_parts = []
                        for idx, p in enumerate(parts):
                            try:
                                if idx == 0:
                                    # 类别ID保持整数
                                    new_parts.append(str(int(float(p))))
                                else:
                                    # 保证严格六位小数（多余截断，少于补零）
                                    float_val = float(p)
                                    formatted = f"{float_val:.6f}"
                                    # 截断多余位数
                                    if '.' in formatted:
                                        int_part, dec_part = formatted.split('.')
                                        dec_part = dec_part[:6].ljust(6, '0')
                                        formatted = f"{int_part}.{dec_part}"
                                    else:
                                        formatted = f"{formatted}.000000"
                                    new_parts.append(formatted)
                            except Exception:
                                new_parts.append(p)
                        lines.append(" ".join(new_parts))
                with open(txt_file, "w") as f:
                    for line in lines:
                        f.write(line + "\n")
            QMessageBox.information(self, "AI 标注完成", message)
            try:
                self.load_annotation_file()
            except Exception:
                pass
        else:
            QMessageBox.critical(self, "AI 标注失败", message)

    def switch_annotation(self, index):
        if 0 <= index < len(self.annotations):
            self.current_annotation = self.annotations[index]
            self.current_category_id = self.current_annotation["category_id"]
            self.category_combo.setCurrentIndex(self.current_category_id)
            self.update_keypoints_list()
            self.update_display()
    
    def refresh_annotation_list(self):
        self.annotation_list.clear()
        for idx, ann in enumerate(self.annotations):
            cname = self.categories[ann["category_id"]]["name"] if 0 <= ann["category_id"] < len(self.categories) else f"class_{ann['category_id']}"
            self.annotation_list.addItem(f"{idx}: {cname}")
        # 保持当前选中
        if self.current_annotation in self.annotations:
            self.annotation_list.setCurrentRow(self.annotations.index(self.current_annotation))

    def parse_keypoints_with_v(self, parts, img_w, img_h):
        kp_data = list(map(float, parts))
        keypoints = []
        # 如果是三元组 (x, y, v)，只取 x, y
        if len(kp_data) % 3 == 0:
            for i in range(0, len(kp_data), 3):
                x, y = kp_data[i], kp_data[i+1]
                if x > 1.0 or y > 1.0:
                    x = x / img_w
                    y = y / img_h
                keypoints.append([x, y, 2])  # v=2，始终可见
        elif len(kp_data) % 2 == 0:
            for i in range(0, len(kp_data), 2):
                x, y = kp_data[i], kp_data[i+1]
                if x > 1.0 or y > 1.0:
                    x = x / img_w
                    y = y / img_h
                v = 2 if (x != 0 and y != 0) else 0
                keypoints.append([x, y, v])
        else:
            pass
        return keypoints

# 运行应用程序
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KeypointAnnotationTool()
    window.show()
    sys.exit(app.exec_())