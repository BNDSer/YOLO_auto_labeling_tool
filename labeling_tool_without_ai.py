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
        
        # 标注数据
        self.annotations = []
        self.current_annotation = None
        self.selected_point_index = -1
        self.dragging = False
        
        # 类别和关键点配置
        self.categories = [
            {
                "name": "person", 
                "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                             "left_wrist", "right_wrist", "left_hip", "right_hip",
                             "left_knee", "right_knee", "left_ankle", "right_ankle"]
            }
        ]
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
        if not self.current_image:
            return
            
        # 获取图像上的实际坐标（考虑缩放）
        pos = event.pos()
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return
            
        # 计算图像在标签中的位置（居中显示）
        img_x = (self.image_label.width() - pixmap.width()) / 2
        img_y = (self.image_label.height() - pixmap.height()) / 2
        
        # 转换为图像坐标
        if pos.x() < img_x or pos.y() < img_y:
            return
            
        x_img = (pos.x() - img_x) / self.scale_factor
        y_img = (pos.y() - img_y) / self.scale_factor
        
        # 检查是否点击了现有的关键点
        if self.current_annotation:
            for i, (x, y, visible) in enumerate(self.current_annotation["keypoints"]):
                if visible > 0:  # 只检查可见的点
                    # 计算屏幕坐标
                    x_screen = x * self.current_image.width() * self.scale_factor + img_x
                    y_screen = y * self.current_image.height() * self.scale_factor + img_y
                    
                    # 检查点击是否在点附近
                    if abs(pos.x() - x_screen) < 10 and abs(pos.y() - y_screen) < 10:
                        self.selected_point_index = i
                        self.dragging = True
                        self.update_display()
                        return
        
        # 如果没有选中现有点，添加新点或创建新标注
        if event.button() == Qt.LeftButton:
            if not self.current_annotation:
                self.start_new_annotation()
            
            # 添加新关键点
            if self.current_annotation and 0 <= self.current_category_id < len(self.categories):
                current_category = self.categories[self.current_category_id]
                
                # 找到第一个不可见的点并设置为当前点
                for i in range(len(current_category["keypoints"])):
                    if i >= len(self.current_annotation["keypoints"]) or self.current_annotation["keypoints"][i][2] == 0:
                        if i >= len(self.current_annotation["keypoints"]):
                            # 扩展关键点列表
                            while len(self.current_annotation["keypoints"]) <= i:
                                self.current_annotation["keypoints"].append([0, 0, 0])
                        
                        # 设置关键点坐标和可见性
                        self.current_annotation["keypoints"][i] = [
                            x_img / self.current_image.width(), 
                            y_img / self.current_image.height(), 
                            2  # 2=可见且已标注
                        ]
                        self.selected_point_index = i
                        self.update_display()
                        break
    
    def image_mouse_move(self, event):
        if self.dragging and self.selected_point_index >= 0 and self.current_annotation:
            # 获取图像上的实际坐标（考虑缩放）
            pos = event.pos()
            pixmap = self.image_label.pixmap()
            if not pixmap:
                return
                
            # 计算图像在标签中的位置（居中显示）
            img_x = (self.image_label.width() - pixmap.width()) / 2
            img_y = (self.image_label.height() - pixmap.height()) / 2
            
            # 转换为图像坐标
            if pos.x() < img_x or pos.y() < img_y:
                return
                
            x_img = max(0, min((pos.x() - img_x) / self.scale_factor, self.current_image.width()))
            y_img = max(0, min((pos.y() - img_y) / self.scale_factor, self.current_image.height()))
            
            # 更新关键点坐标
            self.current_annotation["keypoints"][self.selected_point_index][0] = x_img / self.current_image.width()
            self.current_annotation["keypoints"][self.selected_point_index][1] = y_img / self.current_image.height()
            
            self.update_display()
    
    def image_mouse_release(self, event):
        self.dragging = False
    
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
    
    def update_display(self):
        if not self.current_image:
            return
            
        # 创建显示图像
        display_pixmap = self.current_image.copy()
        painter = QPainter(display_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制所有标注
        for annotation in self.annotations:
            category_id = annotation["category_id"]
            keypoints = annotation["keypoints"]
            
            # 设置颜色（根据类别ID）
            colors = [QColor(0, 255, 0), QColor(255, 0, 0), QColor(0, 0, 255), 
                     QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255)]
            color = colors[category_id % len(colors)]
            
            pen = QPen(color, 2)
            painter.setPen(pen)
            
            # 绘制关键点
            for i, (x, y, visible) in enumerate(keypoints):
                if visible > 0:  # 只绘制可见的点
                    x_pix = x * display_pixmap.width()
                    y_pix = y * display_pixmap.height()
                    
                    # 绘制点
                    painter.drawEllipse(QPoint(int(x_pix), int(y_pix)), 5, 5)
                    
                    # 绘制点编号
                    painter.setFont(QFont("Arial", 10))
                    painter.drawText(QPoint(int(x_pix) + 8, int(y_pix) - 8), str(i))
        
        painter.end()
        
        # 显示图像
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
                        kp_data = list(map(float, parts[5:]))
                        if len(kp_data) % 3 == 0:
                            # 每个关键点为三元组 x,y,v
                            for i in range(0, len(kp_data), 3):
                                x, y, v = kp_data[i], kp_data[i+1], int(kp_data[i+2])
                                keypoints.append([x, y, v])
                        elif len(kp_data) % 2 == 0:
                            # 每个关键点为一对 x,y（没有 v），将 v 设为 2（可见）如果坐标不全为 0
                            for i in range(0, len(kp_data), 2):
                                x, y = kp_data[i], kp_data[i+1]
                                v = 2 if (x != 0 and y != 0) else 0
                                keypoints.append([x, y, v])
                    
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
                    if category_id >= len(self.categories):
                        # 为缺失的类别创建占位
                        for cid in range(len(self.categories), category_id + 1):
                            # 根据读取到的关键点数创建占位关键点名
                            kp_count = max(1, len(normalized_keypoints))
                            placeholder_kps = [f"kp{i}" for i in range(kp_count)]
                            self.categories.append({
                                "name": f"class_{cid}",
                                "keypoints": placeholder_kps
                            })
                        self.update_category_combo()
                        self.status_bar.showMessage(f"检测到未定义类别，已创建占位类别直至 id={category_id}")
                    else:
                        # 如果现有类别的关键点数少于文件中实际的关键点数，则扩展该类别的关键点定义（避免丢失点）
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
                        normalized_keypoints = normalized_keypoints[:needed]
                    
                    annotation = {
                        "category_id": category_id,
                        "bbox": bbox,
                        "keypoints": normalized_keypoints
                    }
                    self.annotations.append(annotation)
            
            if self.annotations:
                self.current_annotation = self.annotations[-1]
                self.current_category_id = self.current_annotation["category_id"]
                # 确保 combo 更新到正确索引
                self.category_combo.setCurrentIndex(self.current_category_id)
            
            self.update_display()
            self.status_bar.showMessage(f"已加载标注文件: {txt_path}")
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
                            kp_data = list(map(float, parts[5:]))
                            if len(kp_data) % 3 == 0:
                                for i in range(0, len(kp_data), 3):
                                    keypoints.append([kp_data[i], kp_data[i+1], int(kp_data[i+2])])
                            elif len(kp_data) % 2 == 0:
                                for i in range(0, len(kp_data), 2):
                                    x, y = kp_data[i], kp_data[i+1]
                                    v = 2 if (x != 0 and y != 0) else 0
                                    keypoints.append([x, y, v])
                        
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
                            # 若现有类别关键点定义少于标签文件实际关键点数，扩展定义以保留点
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
                            normalized_keypoints = normalized_keypoints[:needed]
                        
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

# 运行应用程序
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KeypointAnnotationTool()
    window.show()
    sys.exit(app.exec_())
