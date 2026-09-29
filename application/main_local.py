"""本地版入口：CSV + 本地 Qwen 模型。"""
import os

os.environ["WAFER_PROFILE"] = "local"

from main import main

if __name__ == "__main__":
    main()
