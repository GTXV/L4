# -*- coding: utf-8 -*-
"""
Phone File Manager UI (PyQt5)
--------------------------------
一个精美的手机文件系统可视化界面。你只需要提供后端 API（列表、上传、下载、删除、重命名、新建文件夹等），
将其接入 PhoneAPI 接口即可。此文件不实现具体设备通信逻辑，仅提供 UI 和异步任务管理。

★ 功能亮点
- 左侧目录树（懒加载），右侧文件表格（可排序、筛选）
- 顶部工具栏（返回/前进/上级/刷新/新建/上传/下载/删除/重命名/搜索）
- 面包屑导航、状态栏统计
- 传输队列面板（进度条与状态），支持并发、可扩展
- 拖拽上传（从本地拖文件/文件夹到文件表）
- 深色主题 + 现代化圆角与悬浮效果

★ 接口对接
请实现一个符合 PhoneAPI 的类实例传入 MainWindow(api=你的实现) 即可。
参考 LocalMockAPI（仅用于演示，基于本地文件系统）了解方法签名与进度回调。

运行：
    python phone_file_manager.py
切换为演示模式（使用本地文件系统）：
    在 main() 中将 USE_MOCK = True
"""
import qtawesome as qta

import os
import sys
import typing as T
from dataclasses import dataclass
from datetime import datetime

from PyQt5.QtCore import (
    Qt, QModelIndex, QAbstractTableModel, QSortFilterProxyModel, QSize,
    QRunnable, QObject, pyqtSignal, QThreadPool, QTimer
)
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem, QTableView, QVBoxLayout,
    QWidget, QSplitter, QToolBar, QAction, QFileDialog, QMessageBox, QLineEdit, QLabel,
    QStatusBar, QStyle, QAbstractItemView, QHeaderView, QHBoxLayout, QToolButton,
    QDialog, QInputDialog, QDockWidget, QProgressBar, QMenu
)


from PyQt5.QtCore import QEvent



# ============================
# Data types & API definition
# ============================

@dataclass
class RemoteEntry:
    name: str
    path: str
    is_dir: bool
    size: int = 0
    mtime: T.Optional[float] = None  # epoch seconds


class PhoneAPI:
    """抽象 API：请用你已有的函数实现以下方法。"""

    def list_dir(self, path: str) -> T.List[RemoteEntry]:
        """列出目录，返回 RemoteEntry 列表。path 类似 '/','/DCIM','/Music' 等。"""
        raise NotImplementedError

    def make_dir(self, path: str, name: str) -> str:
        """在 path 下新建文件夹 name，返回新目录的绝对路径。"""
        raise NotImplementedError

    def delete_path(self, path: str) -> None:
        """删除文件或文件夹（递归）。"""
        raise NotImplementedError

    def rename(self, path: str, new_name: str) -> str:
        """重命名，返回新绝对路径。"""
        raise NotImplementedError

    def upload(self, local_path: str, remote_dir: str, progress_cb: T.Callable[[int], None] = None) -> str:
        """上传本地文件/文件夹到 remote_dir。progress_cb 接收 0-100。返回新远程路径。"""
        raise NotImplementedError

    def download(self, remote_path: str, local_dir: str, progress_cb: T.Callable[[int], None] = None) -> str:
        """下载远程文件/文件夹到 local_dir。progress_cb 接收 0-100。返回本地保存路径。"""
        raise NotImplementedError

    def roots(self) -> T.List[str]:
        """返回根目录列表。大多数情况下返回 ['/'] 即可。"""
        return ['/']


# ===============
# Helper & Utils
# ===============

def human_size(n: int) -> str:
    if n is None:
        return ''
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    s = float(n)
    for u in units:
        if s < 1024:
            return f"{s:.2f} {u}" if u != 'B' else f"{int(s)} {u}"
        s /= 1024
    return f"{s:.2f} PB"

def fmt_mtime(t: T.Optional[float]) -> str:
    if not t:
        return ''
    return datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M')


# =======================
# Table model for entries
# =======================

class FileTableModel(QAbstractTableModel):
    HEADERS = ['名称', '大小', '修改时间', '类型']

    def __init__(self):
        super().__init__()
        self._rows: T.List[RemoteEntry] = []

    def set_rows(self, rows: T.List[RemoteEntry]):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return row.name
            elif col == 1:
                return '' if row.is_dir else human_size(row.size)
            elif col == 2:
                return fmt_mtime(row.mtime)
            elif col == 3:
                return '文件夹' if row.is_dir else (os.path.splitext(row.name)[1][1:] or '文件')
        elif role == Qt.DecorationRole and col == 0:
            style = QApplication.style()
            return style.standardIcon(QStyle.SP_DirIcon if row.is_dir else QStyle.SP_FileIcon)
        return None

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        reverse = order == Qt.DescendingOrder
        if column == 0:
            self._rows.sort(key=lambda r: (not r.is_dir, r.name.lower()), reverse=reverse)
        elif column == 1:
            self._rows.sort(key=lambda r: (r.is_dir, r.size), reverse=reverse)
        elif column == 2:
            self._rows.sort(key=lambda r: (r.is_dir, r.mtime or 0), reverse=reverse)
        elif column == 3:
            self._rows.sort(key=lambda r: (r.is_dir, os.path.splitext(r.name)[1].lower()), reverse=reverse)
        self.layoutChanged.emit()

    def entry_at(self, idx: int) -> RemoteEntry:
        return self._rows[idx]

    def entries(self) -> T.List[RemoteEntry]:
        return self._rows[:]


class NameFilterProxy(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self._pattern = ''

    def setPattern(self, text: str):
        self._pattern = text.lower().strip()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        if not self._pattern:
            return True
        idx = self.sourceModel().index(source_row, 0, source_parent)
        name = self.sourceModel().data(idx, Qt.DisplayRole) or ''
        return self._pattern in str(name).lower()


# ====================
# Threading utilities
# ====================

class WorkerSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(Exception)
    finished = pyqtSignal()
    progress = pyqtSignal(int)  # 0-100


class Worker(QRunnable):
    def __init__(self, fn: T.Callable, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # 若提供 progress_cb 参数，则注入发射器
        if 'progress_cb' in self.fn.__code__.co_varnames and 'progress_cb' not in self.kwargs:
            self.kwargs['progress_cb'] = self.signals.progress.emit

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.error.emit(e)
        else:
            self.signals.result.emit(res)
        finally:
            self.signals.finished.emit()


# =============
# Main Window
# =============

class MainWindow(QMainWindow):
    def __init__(self, api: PhoneAPI, start_path: str = '/'):
        super().__init__()
        self.api = api
        self.thread_pool = QThreadPool.globalInstance()

        self.history: T.List[str] = []
        self.history_pos = -1
        self.current_path = start_path

        self.setWindowTitle("Phone File Manager")
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.resize(1200, 720)

        self._build_ui()
        self._apply_theme()
        self._connect_signals()

        # 初始化
        self.navigate_to(start_path, add_history=True)

    # ----- UI -----
    def _build_ui(self):
        # Toolbar
        tb = QToolBar()
        tb.setIconSize(QSize(22, 22))
        self.addToolBar(tb)

        ACCENT = "#2563eb"   # 主题色（可换），浅色背景下更好看
        ICONC  = "#374151"   # 图标描边色（可换）

        self.act_back = QAction(qta.icon("mdi.arrow-left-circle-outline", color=ICONC), "后退", self)
        self.act_forward = QAction(qta.icon("mdi.arrow-right-circle-outline", color=ICONC), "前进", self)
        self.act_up = QAction(qta.icon("mdi.arrow-up-circle-outline", color=ICONC), "上级", self)
        self.act_refresh = QAction(qta.icon("mdi.refresh", color=ICONC), "刷新", self)

        self.act_new_folder = QAction(qta.icon("mdi.folder-plus-outline", color=ICONC), "新建文件夹", self)
        self.act_upload = QAction(qta.icon("mdi.cloud-upload-outline", color=ACCENT), "上传", self)
        self.act_download = QAction(qta.icon("mdi.cloud-download-outline", color=ACCENT), "下载", self)
        self.act_delete = QAction(qta.icon("mdi.trash-can-outline", color="#dc2626"), "删除", self)
        self.act_rename = QAction(qta.icon("mdi.pencil-outline", color=ICONC), "重命名", self)

        tb.addActions([self.act_back, self.act_forward, self.act_up, self.act_refresh])
        tb.addSeparator()
        tb.addActions([self.act_new_folder, self.act_upload, self.act_download, self.act_delete, self.act_rename])
        tb.addSeparator()
        '''
        tb.addWidget(QLabel(" 搜索："))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("按名称筛选当前目录…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setFixedWidth(240)
        tb.addWidget(self.search_edit)
        '''
        # Breadcrumbs
        crumb_bar = QWidget()
        crumb_layout = QHBoxLayout(crumb_bar)
        crumb_layout.setContentsMargins(6, 4, 6, 4)   # 更紧凑
        crumb_layout.setSpacing(6)

        self.path_label = QLabel("当前位置: /")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setStyleSheet("color: #666;")
        crumb_layout.addWidget(self.path_label)

        crumb_bar.setFixedHeight(28)
        crumb_layout = QHBoxLayout(crumb_bar)
        crumb_layout.setContentsMargins(8, 8, 8, 8)
        crumb_layout.setSpacing(6)
        self.crumb_layout = crumb_layout

        # Left tree + Right table
        self.dir_tree = QTreeWidget()
        self.dir_tree.setHeaderHidden(True)
        self.dir_tree.setExpandsOnDoubleClick(True)
        self.dir_tree.setAlternatingRowColors(True)
        self.dir_tree.setAnimated(True)

        for root in self.api.roots():
            item = QTreeWidgetItem([root])
            item.setData(0, Qt.UserRole, root)
            item.setIcon(0, self.style().standardIcon(QStyle.SP_DirIcon))
            item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
            self.dir_tree.addTopLevelItem(item)

        self.file_table = QTableView()

        self.file_table.setAcceptDrops(True)
        self.file_table.viewport().setAcceptDrops(True)
        self.file_table.viewport().installEventFilter(self)

        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.file_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_table.setSortingEnabled(True)
        self.file_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.file_table.verticalHeader().setVisible(False)
        self.file_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.file_table.setAlternatingRowColors(True)
        self.file_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_table.setAcceptDrops(True)
        self.file_table.setDragDropMode(QAbstractItemView.DropOnly)
        self.file_table.setDropIndicatorShown(True)

        self.model = FileTableModel()
        self.proxy = NameFilterProxy()
        self.proxy.setSourceModel(self.model)
        self.file_table.setModel(self.proxy)

        # Splitter
        splitter = QSplitter()
        splitter.addWidget(self.dir_tree)
        splitter.addWidget(self.file_table)
        splitter.setStretchFactor(1, 1)

        # Transfer dock
        self.transfer_dock = QDockWidget("传输队列", self)
        self.transfer_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        dock_widget = QWidget()
        dock_layout = QVBoxLayout(dock_widget)
        dock_layout.setContentsMargins(8, 8, 8, 8)
        dock_layout.setSpacing(8)

        self.queue_list = QTreeWidget()
        self.queue_list.setHeaderLabels(["任务", "路径", "进度", "状态"])
        self.queue_list.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.queue_list.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.queue_list.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.queue_list.header().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        dock_layout.addWidget(self.queue_list)
        self.transfer_dock.setWidget(dock_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.transfer_dock)

        # Central
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(crumb_bar)
        lay.addWidget(splitter)
        self.setCentralWidget(central)

        # Status bar
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_count = QLabel("0 项")
        sb.addPermanentWidget(self.lbl_count)

    def eventFilter(self, obj, event):
    # 仅处理“文件表”的拖拽
        if obj is self.file_table.viewport():
            if event.type() == QEvent.DragEnter or event.type() == QEvent.DragMove:
                md = event.mimeData()
                if md and md.hasUrls():
                    event.acceptProposedAction()
                    return True  # 我们接管
            elif event.type() == QEvent.Drop:
                md = event.mimeData()
                if md and md.hasUrls():
                    # 目标目录：默认是当前路径
                    dst_dir = self.current_path

                    # （可选）如果在一个“目录行”上方放下，则把那个目录当作目标
                    pos = event.pos()
                    idx = self.file_table.indexAt(pos)
                    if idx.isValid():
                        src_idx = self.proxy.mapToSource(idx)
                        entry = self.model.entry_at(src_idx.row())
                        if entry.is_dir:
                            dst_dir = entry.path

                    for url in md.urls():
                        local = url.toLocalFile()
                        if local:
                            self.enqueue_transfer(task="上传", src=local, dst=dst_dir, is_upload=True)

                    event.acceptProposedAction()
                    return True
        # 其它情况交回默认处理
        return super().eventFilter(obj, event)



    def _apply_theme(self):
        QApplication.setStyle("Fusion")
        # 现代深色主题（包含圆角、悬浮、选中高亮）
        '''
        self.setStyleSheet("""
            QMainWindow { background: #14151a; }
            QToolBar { background: #151821; border: none; padding: 6px; }
            QToolButton { color: #eaeef7; padding: 6px 10px; border-radius: 10px; }
            QToolButton:hover { background: #24283b; }
            QLineEdit { background: #0f1117; border: 1px solid #2d3153; border-radius: 10px; color: #eaeef7; padding: 6px 10px; }
            QTreeWidget, QTableView { background: #0f1117; alternate-background-color: #12141c; color: #eaeef7;
                border: 1px solid #1f243c; border-radius: 12px; }
            QHeaderView::section { background: #151821; color: #cfd6f6; padding: 8px; border: none; }
            QStatusBar { background: #151821; color: #cfd6f6; }
            QProgressBar { border: 1px solid #1f243c; border-radius: 8px; background: #11131a; text-align: center; color: #eaeef7; }
            QProgressBar::chunk { background-color: #7c5cff; border-radius: 8px; }
            QDockWidget::title { text-align: left; background: #151821; padding: 8px; }
            QMenu { background: #151821; color: #eaeef7; border: 1px solid #1f243c; }
            QMenu::item:selected { background: #24283b; }
            QLabel { color: #cfd6f6; }
            QToolTip { color: #eaeef7; background-color: #24283b; border: 1px solid #2d3153; }
            QTableView::item:selected, QTreeWidget::item:selected { background: #2a2f4a; }
        """)
        '''
    def _connect_signals(self):
        self.act_back.triggered.connect(self.go_back)
        self.act_forward.triggered.connect(self.go_forward)
        self.act_up.triggered.connect(self.go_up)
        self.act_refresh.triggered.connect(lambda: self.navigate_to(self.current_path, add_history=False))

        self.act_new_folder.triggered.connect(self.create_folder)
        self.act_upload.triggered.connect(self.upload_files)
        self.act_download.triggered.connect(self.download_selected)
        self.act_delete.triggered.connect(self.delete_selected)
        self.act_rename.triggered.connect(self.rename_selected)

        self.dir_tree.itemExpanded.connect(self.on_tree_expand)
        self.dir_tree.itemClicked.connect(self.on_tree_click)

        #self.search_edit.textChanged.connect(self.proxy.setPattern)
        self.file_table.doubleClicked.connect(self.on_table_double_click)
        self.file_table.customContextMenuRequested.connect(self.on_table_context_menu)

    # ----- Navigation -----
    '''
    def build_breadcrumbs(self, path: str):
        # 清空
        while self.crumb_layout.count():
            w = self.crumb_layout.takeAt(0).widget()
            if w:
                w.deleteLater()

        parts = [p for p in path.split('/') if p]
        accum = '/'
        def add_btn(text, pth):
            btn = QToolButton()
            btn.setText(text)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.clicked.connect(lambda: self.navigate_to(pth, add_history=True))
            self.crumb_layout.addWidget(btn)

        add_btn('根目录', '/')
        for i, seg in enumerate(parts):
            accum = (accum.rstrip('/') + '/' + seg).replace('//', '/')
            arrow = QLabel('›')
            self.crumb_layout.addWidget(arrow)
            add_btn(seg, accum)

        self.crumb_layout.addStretch(1)
    '''
    def build_breadcrumbs(self, path: str):
        self.path_label.setText(f"当前位置: {path}")
        self.path_label.setToolTip(path) 
    def go_back(self):
        if self.history_pos > 0:
            self.history_pos -= 1
            self.navigate_to(self.history[self.history_pos], add_history=False)

    def go_forward(self):
        if self.history_pos < len(self.history) - 1:
            self.history_pos += 1
            self.navigate_to(self.history[self.history_pos], add_history=False)

    def go_up(self):
        if self.current_path == '/':
            return
        parent = os.path.dirname(self.current_path.rstrip('/')) or '/'
        self.navigate_to(parent, add_history=True)

    def navigate_to(self, path: str, add_history: bool = True):
        self.current_path = path
        self.build_breadcrumbs(path)
        self.statusBar().showMessage(f"加载中：{path} …", 2000)

        worker = Worker(self.api.list_dir, path)
        worker.signals.result.connect(self._on_dir_loaded)
        worker.signals.error.connect(self._on_error)
        self.thread_pool.start(worker)

        if add_history:
            # 修剪 forward 历史
            self.history = self.history[: self.history_pos + 1]
            self.history.append(path)
            self.history_pos = len(self.history) - 1
    '''
    def _on_dir_loaded(self, entries: T.List[RemoteEntry]):
        # 目录优先显示
        entries = sorted(entries, key=lambda r: (not r.is_dir, r.name.lower()))
        self.model.set_rows(entries)
        self.lbl_count.setText(f"{len(entries)} 项")
        # 更新树上选中高亮
        self._select_tree_path(self.current_path)
    '''
    def _on_dir_loaded(self, entries):
        entries = sorted(entries, key=lambda r: (not r.is_dir, r.name.lower()))
        self.model.set_rows(entries)
        self.lbl_count.setText(f"{len(entries)} 项")

    def _on_error(self, e: Exception):
        QMessageBox.critical(self, "错误", str(e))

    # ----- Tree -----
    def on_tree_expand(self, item: QTreeWidgetItem):
        path = item.data(0, Qt.UserRole)
        # 若已加载过则跳过
        if item.childCount() > 0 and item.child(0).data(0, Qt.UserRole) != '__placeholder__':
            return
        # 清空占位
        item.takeChildren()

        def after(entries: T.List[RemoteEntry]):
            for ent in entries:
                if ent.is_dir:
                    child = QTreeWidgetItem([ent.name])
                    child.setData(0, Qt.UserRole, ent.path)
                    child.setIcon(0, self.style().standardIcon(QStyle.SP_DirIcon))
                    child.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
                    item.addChild(child)

        worker = Worker(self.api.list_dir, path)
        worker.signals.result.connect(after)
        worker.signals.error.connect(self._on_error)
        self.thread_pool.start(worker)

    def _select_tree_path(self, path: str):
        # 尝试展开到对应路径
        segments = [s for s in path.split('/') if s]
        if not segments:
            # 选中根
            if self.dir_tree.topLevelItemCount():
                self.dir_tree.setCurrentItem(self.dir_tree.topLevelItem(0))
            return

        def find_child(parent: QTreeWidgetItem, name: str) -> T.Optional[QTreeWidgetItem]:
            for i in range(parent.childCount()):
                if parent.child(i).text(0) == name:
                    return parent.child(i)
            return None

        if self.dir_tree.topLevelItemCount() == 0:
            return
        cur = self.dir_tree.topLevelItem(0)
        cur.setExpanded(True)
        for seg in segments:
            # 展开并懒加载
            self.on_tree_expand(cur)
            child = find_child(cur, seg)
            if not child:
                break
            child.setExpanded(True)
            cur = child
        self.dir_tree.setCurrentItem(cur)

    def on_tree_click(self, item: QTreeWidgetItem, column: int):
        path = item.data(0, Qt.UserRole)
        self.navigate_to(path, add_history=True)

    # ----- Table / actions -----
    def on_table_double_click(self, index: QModelIndex):
        src_idx = self.proxy.mapToSource(index)
        entry = self.model.entry_at(src_idx.row())
        if entry.is_dir:
            self.navigate_to(entry.path, add_history=True)

    def on_table_context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction(self.act_upload)
        menu.addAction(self.act_download)
        menu.addAction(self.act_new_folder)
        menu.addSeparator()
        menu.addAction(self.act_rename)
        menu.addAction(self.act_delete)
        menu.exec_(self.file_table.viewport().mapToGlobal(pos))

    def selected_entries(self) -> T.List[RemoteEntry]:
        rows = []
        for idx in self.file_table.selectionModel().selectedRows():
            src = self.proxy.mapToSource(idx)
            rows.append(self.model.entry_at(src.row()))
        return rows

    def create_folder(self):
        name, ok = QInputDialog.getText(self, "新建文件夹", "输入名称：")
        if not ok or not name.strip():
            return
        def done(new_path: str):
            self.navigate_to(self.current_path, add_history=False)
        worker = Worker(self.api.make_dir, self.current_path, name.strip())
        worker.signals.result.connect(done)
        worker.signals.error.connect(self._on_error)
        self.thread_pool.start(worker)

    def rename_selected(self):
        sels = self.selected_entries()
        if len(sels) != 1:
            QMessageBox.information(self, "提示", "请选择 1 个项目进行重命名。")
            return
        entry = sels[0]
        new_name, ok = QInputDialog.getText(self, "重命名", "输入新名称：", text=entry.name)
        if not ok or not new_name.strip() or new_name == entry.name:
            return
        def done(new_path: str):
            self.navigate_to(self.current_path, add_history=False)
        worker = Worker(self.api.rename, entry.path, new_name.strip())
        worker.signals.result.connect(done)
        worker.signals.error.connect(self._on_error)
        self.thread_pool.start(worker)

    def upload_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择要上传的文件")
        if not paths:
            # 也允许选择文件夹
            dir_path = QFileDialog.getExistingDirectory(self, "选择要上传的文件夹")
            if dir_path:
                paths = [dir_path]
        for p in paths:
            self.enqueue_transfer(task="上传", src=p, dst=self.current_path, is_upload=True)

    def download_selected(self):
        sels = self.selected_entries()
        if not sels:
            QMessageBox.information(self, "提示", "请先选择要下载的文件/文件夹。")
            return
        local_dir = QFileDialog.getExistingDirectory(self, "选择保存到的本地文件夹")
        if not local_dir:
            return
        for e in sels:
            self.enqueue_transfer(task="下载", src=e.path, dst=local_dir, is_upload=False)

    def delete_selected(self):
        sels = self.selected_entries()
        if not sels:
            return
        names = "\n".join(e.name for e in sels)
        if QMessageBox.question(self, "确认删除", f"确定删除以下项目？\n\n{names}") != QMessageBox.Yes:
            return
        def after(_):
            self.navigate_to(self.current_path, add_history=False)
        for e in sels:
            worker = Worker(self.api.delete_path, e.path)
            worker.signals.error.connect(self._on_error)
            worker.signals.finished.connect(lambda: after(None))
            self.thread_pool.start(worker)

    # ----- Transfer queue -----
    def enqueue_transfer(self, task: str, src: str, dst: str, is_upload: bool):
        #item = QTreeWidgetItem([task, src if is_upload else dst, "0%", "排队中"])
        item = QTreeWidgetItem([task, (dst if is_upload else dst), "0%", "排队中"])
        pb = QProgressBar()
        pb.setValue(0)
        self.queue_list.addTopLevelItem(item)
        self.queue_list.setItemWidget(item, 2, pb)

        '''
        def on_result(_res):
            item.setText(3, "完成")
            pb.setValue(100)
            if is_upload:
                # 上传完成后刷新当前目录
                QTimer.singleShot(300, lambda: self.navigate_to(self.current_path, add_history=False))
        '''
        dst_dir = dst  # ← 在 enqueue_transfer 开头就捕获
        def on_result(_res):
            item.setText(3, "完成")
            pb.setValue(100)
            if is_upload:
                QTimer.singleShot(300, lambda: self.navigate_to(dst_dir, add_history=False))
        def on_error(e: Exception):
            item.setText(3, f"失败: {e}")
            pb.setValue(0)

        def on_progress(v: int):
            pb.setValue(max(0, min(100, int(v))))
            item.setText(3, "传输中")

        fn = self.api.upload if is_upload else self.api.download
        args = (src, dst)
        worker = Worker(fn, *args)
        worker.signals.result.connect(on_result)
        worker.signals.error.connect(on_error)
        worker.signals.progress.connect(on_progress)
        self.thread_pool.start(worker)

    # 支持拖拽上传
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        for u in urls:
            local = u.toLocalFile()
            if local:
                self.enqueue_transfer(task="上传", src=local, dst=self.current_path, is_upload=True)



# ====================
# Local Mock (demo)
# ====================
class LocalMockAPI(PhoneAPI):
    """演示用：用本地文件系统模拟设备。你不需要在生产中使用它。"""

    def __init__(self, root: str = None):
        self._root = root or os.path.expanduser("~")

    def _real(self, remote_path: str) -> str:
        if remote_path in ('', '/'):
            return self._root
        if remote_path.startswith('/'):
            remote_path = remote_path[1:]
        return os.path.join(self._root, remote_path)

    def roots(self) -> T.List[str]:
        return ['/']

    def list_dir(self, path: str) -> T.List[RemoteEntry]:
        rp = self._real(path)
        rows: T.List[RemoteEntry] = []
        for name in sorted(os.listdir(rp), key=lambda s: s.lower()):
            p = os.path.join(rp, name)
            st = os.stat(p)
            rows.append(RemoteEntry(
                name=name,
                path=os.path.join(path.rstrip('/'), name) if path != '/' else '/' + name,
                is_dir=os.path.isdir(p),
                size=st.st_size if os.path.isfile(p) else 0,
                mtime=st.st_mtime
            ))
        return rows

    def make_dir(self, path: str, name: str) -> str:
        rp = self._real(path)
        newp = os.path.join(rp, name)
        os.makedirs(newp, exist_ok=True)
        return os.path.join(path.rstrip('/'), name) if path != '/' else '/' + name

    def delete_path(self, path: str) -> None:
        rp = self._real(path)
        if os.path.isdir(rp):
            import shutil
            shutil.rmtree(rp)
        else:
            os.remove(rp)

    def rename(self, path: str, new_name: str) -> str:
        rp = self._real(path)
        parent = os.path.dirname(rp)
        newp = os.path.join(parent, new_name)
        os.rename(rp, newp)
        parent_remote = os.path.dirname(path.rstrip('/')) or '/'
        return os.path.join(parent_remote.rstrip('/'), new_name) if parent_remote != '/' else '/' + new_name

    def upload(self, local_path: str, remote_dir: str, progress_cb=None) -> str:
        import shutil
        rp = self._real(remote_dir)
        base = os.path.basename(local_path.rstrip(os.sep))
        dest = os.path.join(rp, base)
        # 粗略进度模拟
        if os.path.isdir(local_path):
            shutil.copytree(local_path, dest)
        else:
            with open(local_path, 'rb') as fsrc, open(dest, 'wb') as fdst:
                total = os.path.getsize(local_path) or 1
                copied = 0
                chunk = 1024 * 512
                while True:
                    buf = fsrc.read(chunk)
                    if not buf:
                        break
                    fdst.write(buf)
                    copied += len(buf)
                    if progress_cb:
                        progress_cb(int(copied * 100 / total))
        if progress_cb:
            progress_cb(100)
        return dest

    def download(self, remote_path: str, local_dir: str, progress_cb=None) -> str:
        import shutil
        src = self._real(remote_path)
        base = os.path.basename(src)
        dest = os.path.join(local_dir, base)
        # 粗略进度模拟
        if os.path.isdir(src):
            shutil.copytree(src, dest)
            if progress_cb:
                progress_cb(100)
        else:
            total = os.path.getsize(src) or 1
            copied = 0
            chunk = 1024 * 512
            with open(src, 'rb') as fsrc, open(dest, 'wb') as fdst:
                while True:
                    buf = fsrc.read(chunk)
                    if not buf:
                        break
                    fdst.write(buf)
                    copied += len(buf)
                    if progress_cb:
                        progress_cb(int(copied * 100 / total))
            if progress_cb:
                progress_cb(100)
        return dest


# =========
# main
# =========

def main():
    app = QApplication(sys.argv)

    # 将 USE_MOCK = False 并传入你的真实 API 实现实例
    USE_MOCK = True
    if USE_MOCK:
        api = LocalMockAPI()
    else:
        class YourAPI(PhoneAPI):
            pass
        api = YourAPI()  # TODO: 替换为你的实现

    win = MainWindow(api, start_path='/')
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()