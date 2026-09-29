"""服务器版入口：Qdrant + vLLM Embedding 服务。"""
import os

os.environ["WAFER_PROFILE"] = "server"

from main import main

if __name__ == "__main__":
    main()
