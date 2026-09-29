"""DINOv2半导体缺陷分类与图像特征API包。

本文件使 ``inference_api`` 成为可导入的Python包。Uvicorn导入
``inference_api.server:app`` 时会先执行本文件；这里不加载模型、不启动端口，
实际HTTP应用和模型服务分别定义在 ``server.py`` 与 ``model_service.py``。
"""
