# data_process/_init_path.py
import sys
import os

# ===============================
# 1️⃣ 工程根目录（本文件所在目录）
# ===============================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# ===============================
# 2️⃣ 加入 Python 模块搜索路径
# ===============================
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ===============================
# 3️⃣ 可选：调试信息（稳定后可删）
# ===============================
print(f"[Path Init] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[Path Init] my_lib found  = {os.path.exists(os.path.join(PROJECT_ROOT, 'my_lib'))}")
