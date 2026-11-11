import time
import requests
from pinecone import Pinecone, ServerlessSpec
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None

EMBEDDING_DIMENSIONS = {
    "all-minilm-l6-v2": 384,
    "liquid/lfm-2.5-embedding-350m:free": 1024,
}


class OpenRouterEmbeddings(Embeddings):
    """Turns text into number lists (embeddings) using OpenRouter."""

    def __init__(self, model: str, api_key: str, batch_size: int = 16):
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.url = "https://openrouter.ai/api/v1/embeddings"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(4):
            response = requests.post(
                self.url,
                headers=headers,
                json={"model": self.model, "input": texts},
                timeout=60,
            )
            if response.status_code == 429:  # free model busy, wait and retry
                time.sleep(2 * (attempt + 1))
                continue
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            #return [item["embedding"] for item in data]
            return [[float(x) for x in item["embedding"]] for item in data]
        raise RuntimeError("OpenRouter rate limit: too many requests, try again soon")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for i in range(0, len(texts), self.batch_size):
            results.extend(self._embed(texts[i:i + self.batch_size]))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "all-minilm" in normalized:
        return 384

    raise ValueError(
        f"Unsupported embedding model '{name}' for Pinecone. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")
        _embeddings = OpenRouterEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openrouter_api_key,
        )
    return _embeddings


def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    names = [x["name"] for x in pc.list_indexes()]

    if settings.pinecone_index_name in names:
        index_info = pc.describe_index(settings.pinecone_index_name)
        current_dimension = getattr(index_info, "dimension", None)
        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        if current_dimension is not None and current_dimension != desired_dimension:
            pc.delete_index(name=settings.pinecone_index_name)
            while settings.pinecone_index_name in [x["name"] for x in pc.list_indexes()]:
                time.sleep(1)

    if settings.pinecone_index_name not in [x["name"] for x in pc.list_indexes()]:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(settings.pinecone_index_name)


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})


def add_documents(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)