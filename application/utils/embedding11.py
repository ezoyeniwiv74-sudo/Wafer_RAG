"""
向量生成模块 - 调用本地 OpenAI 兼容服务生成文本 Embedding
"""

import json
import time
from urllib.request import Request, urlopen
from openai import OpenAI
from config import (VLLM_BASE_URL, VLLM_API_KEY, MODEL_NAME, RUNTIME_PROFILE,
                    LOCAL_MODEL_PATH, LOCAL_EMBED_BATCH_SIZE, LOCAL_EMBED_MAX_LENGTH,
                    OLLAMA_EMBED_KEEP_ALIVE)


class EmbeddingGenerator:
    """向量生成器"""

    def __init__(self):
        self.client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY
        )
        self.model_name = MODEL_NAME

    def _local_embeddings(self, values):
        """Call a localhost OpenAI-compatible service without proxy discovery."""
        base_url = VLLM_BASE_URL.rstrip("/")
        # Ollama's native endpoint can unload the embedding runner before the
        # report model starts, so both models do not occupy VRAM at once.
        if base_url.endswith("/v1") and ":" in self.model_name:
            url = base_url[:-3] + "/api/embed"
            request_body = {"input": values, "model": self.model_name,
                            "keep_alive": OLLAMA_EMBED_KEEP_ALIVE}
        else:
            url = base_url + "/embeddings"
            request_body = {"input": values, "model": self.model_name}
        body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        request = Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLLM_API_KEY}",
        })
        with urlopen(request, timeout=600) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if "embeddings" in payload:
            return payload["embeddings"]
        return [item["embedding"] for item in sorted(payload.get("data", []), key=lambda item: item["index"])]

    def get_embedding(self, text: str, is_query: bool = False) -> list:
        """
        调用本地服务生成文本的向量嵌入

        Args:
            text: 输入文本
            is_query: 是否为查询文本（会添加查询指令前缀）

        Returns:
            向量列表
        """
        text = text.strip()
        if not text:
            raise ValueError("输入文本不能为空")

        # 针对 Qwen3-Embedding 拼接指令引导
        if is_query:
            input_text = f"为晶圆异常分析与检测任务检索相关文档：{text}"
        else:
            input_text = f"晶圆数据记录：{text}"

        try:
            if VLLM_BASE_URL.startswith(("http://127.0.0.1", "http://localhost")):
                vectors = self._local_embeddings([input_text])
                if not vectors:
                    raise RuntimeError("Local embedding service returned no data")
                return vectors[0]
            response = self.client.embeddings.create(input=[input_text], model=self.model_name)
        except Exception as e:
            raise RuntimeError(
                "本地 Embedding 服务调用失败，请检查：\n"
                "1. vLLM 服务是否已启动；\n"
                "2. 服务地址是否正确；\n"
                "3. 模型名称是否与服务启动时一致\n"
                f"原始错误：{e}"
            ) from e

        if not response.data:
            raise RuntimeError("本地服务未返回任何 Embedding 数据")

        vector = response.data[0].embedding

        if not vector:
            raise RuntimeError("本地服务返回的 Embedding 向量为空")

        return vector

    def get_embedding_with_time(self, text: str, is_query: bool = False) -> tuple:
        """
        生成向量并返回耗时

        Returns:
            (vector, elapsed_ms)
        """
        start_time = time.perf_counter()
        vector = self.get_embedding(text, is_query)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return vector, elapsed_ms

    def get_embeddings(self, texts, batch_size=32) -> list:
        """Generate document embeddings in server-side batches."""
        values = [str(text).strip() for text in texts]
        if any(not value for value in values):
            raise ValueError("输入文本不能为空")
        result = []
        for start in range(0, len(values), batch_size):
            batch = [f"晶圆数据记录：{text}" for text in values[start:start + batch_size]]
            if VLLM_BASE_URL.startswith(("http://127.0.0.1", "http://localhost")):
                result.extend(self._local_embeddings(batch))
            else:
                response = self.client.embeddings.create(input=batch, model=self.model_name)
                result.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return result


class LocalEmbeddingGenerator:
    """本地模型实现，与远程 EmbeddingGenerator 暴露相同接口。"""

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(str(LOCAL_MODEL_PATH), trust_remote_code=True)
            self.backend = "sentence_transformers"
        except ImportError:
            # 项目已有 transformers/torch 时无需额外安装 sentence-transformers。
            import torch
            from transformers import AutoModel, AutoTokenizer
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_PATH), padding_side="left")
            self.model = AutoModel.from_pretrained(str(LOCAL_MODEL_PATH), trust_remote_code=True)
            self.model.eval()
            self.backend = "transformers"

    def get_embedding(self, text: str, is_query: bool = False) -> list:
        text = text.strip()
        if not text:
            raise ValueError("输入文本不能为空")
        prefix = "为晶圆异常分析与检测任务检索相关文档：" if is_query else "晶圆数据记录："
        input_text = prefix + text
        if self.backend == "sentence_transformers":
            vector = self.model.encode(input_text, normalize_embeddings=True)
            return vector.tolist()
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=LOCAL_EMBED_MAX_LENGTH)
        with self.torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            vector = hidden[:, -1, :]
            vector = self.torch.nn.functional.normalize(vector, p=2, dim=1)
        return vector[0].cpu().float().tolist()

    def get_embeddings(self, texts, batch_size=LOCAL_EMBED_BATCH_SIZE):
        """批量生成文档向量，显著减少本地首次建索引时间。"""
        prefixed = ["晶圆数据记录：" + str(text).strip() for text in texts]
        if self.backend == "sentence_transformers":
            return self.model.encode(prefixed, batch_size=batch_size,
                                     normalize_embeddings=True).tolist()
        vectors = []
        for start in range(0, len(prefixed), batch_size):
            batch = prefixed[start:start + batch_size]
            print(f"本地向量索引: {min(start + len(batch), len(prefixed))}/{len(prefixed)}", end="\r", flush=True)
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=LOCAL_EMBED_MAX_LENGTH)
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                value = hidden[:, -1, :]
                value = self.torch.nn.functional.normalize(value, p=2, dim=1)
            vectors.extend(value.cpu().float().tolist())
        print()
        return vectors

    def get_embedding_with_time(self, text: str, is_query: bool = False) -> tuple:
        started = time.perf_counter()
        vector = self.get_embedding(text, is_query)
        return vector, (time.perf_counter() - started) * 1000


# 单例模式
_embedding_generator = None


def get_embedding_generator() -> EmbeddingGenerator:
    """获取 Embedding 生成器单例"""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = (LocalEmbeddingGenerator() if RUNTIME_PROFILE == "local"
                                else EmbeddingGenerator())
    return _embedding_generator
