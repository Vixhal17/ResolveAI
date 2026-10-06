import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

settings = get_settings()


# ---------------------------------------------------------
# Global cached objects
# ---------------------------------------------------------

_embeddings = None
_vectorstore = None


# ---------------------------------------------------------
# Embedding dimensions
# ---------------------------------------------------------

EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-minilm-l6-v2": 384,
}


# ---------------------------------------------------------
# Get embedding dimension
# ---------------------------------------------------------

def get_embedding_dimension(model_name: str | None = None) -> int:
    """
    Return the vector dimension for the configured
    Hugging Face embedding model.
    """

    name = (
        model_name
        or settings.embedding_model
        or ""
    ).strip()

    if not name:
        raise RuntimeError(
            "EMBEDDING_MODEL is not configured"
        )

    normalized = name.lower()

    # Exact / normalized lookup
    for model, dimension in EMBEDDING_DIMENSIONS.items():
        if normalized == model.lower():
            return dimension

    # Flexible matching
    if "all-minilm-l6-v2" in normalized:
        return 384

    raise ValueError(
        f"Unsupported embedding model: '{name}'. "
        "Add its dimension to EMBEDDING_DIMENSIONS."
    )


# ---------------------------------------------------------
# Hugging Face Embeddings
# ---------------------------------------------------------

def get_embeddings():
    """
    Create and cache Hugging Face embeddings.
    """

    global _embeddings

    if _embeddings is None:

        model_name = settings.embedding_model

        if not model_name:
            raise RuntimeError(
                "EMBEDDING_MODEL is missing"
            )

        print(
            f"Loading embedding model: {model_name}"
        )

        _embeddings = HuggingFaceEmbeddings(
            model_name=model_name,

            model_kwargs={
                "device": "cpu",
            },

            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        print(
            "Embedding model loaded successfully."
        )

    return _embeddings


# ---------------------------------------------------------
# Pinecone Index
# ---------------------------------------------------------

def ensure_index():
    """
    Create the Pinecone index if it does not exist.

    Also verifies that the existing index has the
    correct embedding dimension.
    """

    if not settings.pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is missing"
        )

    index_name = settings.pinecone_index_name

    if not index_name:
        raise RuntimeError(
            "PINECONE_INDEX_NAME is missing"
        )

    # Get embedding dimension
    desired_dimension = get_embedding_dimension()

    print(
        f"Embedding model: {settings.embedding_model}"
    )

    print(
        f"Embedding dimension: {desired_dimension}"
    )

    # Initialize Pinecone
    pc = Pinecone(
        api_key=settings.pinecone_api_key
    )

    # Get existing indexes
    existing_indexes = [
        index["name"]
        for index in pc.list_indexes()
    ]

    # -----------------------------------------------------
    # Index already exists
    # -----------------------------------------------------

    if index_name in existing_indexes:

        print(
            f"Pinecone index '{index_name}' already exists."
        )

        index_info = pc.describe_index(
            index_name
        )

        current_dimension = getattr(
            index_info,
            "dimension",
            None
        )

        # Handle dictionary response as well
        if (
            current_dimension is None
            and isinstance(index_info, dict)
        ):
            current_dimension = index_info.get(
                "dimension"
            )

        # Check dimension
        if (
            current_dimension is not None
            and current_dimension != desired_dimension
        ):

            raise RuntimeError(
                "\nPinecone dimension mismatch!\n"
                f"Existing index dimension: "
                f"{current_dimension}\n"
                f"Required dimension: "
                f"{desired_dimension}\n\n"
                "Your embedding model is:\n"
                f"{settings.embedding_model}\n\n"
                "You must delete/recreate the Pinecone "
                "index with dimension "
                f"{desired_dimension}."
            )

        print(
            f"Pinecone dimension verified: "
            f"{current_dimension}"
        )

    # -----------------------------------------------------
    # Create index
    # -----------------------------------------------------

    else:

        print(
            f"Creating Pinecone index '{index_name}'..."
        )

        pc.create_index(
            name=index_name,

            dimension=desired_dimension,

            metric="cosine",

            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        # Wait until index is ready
        print(
            "Waiting for Pinecone index to become ready..."
        )

        while True:

            index_info = pc.describe_index(
                index_name
            )

            status = index_info.status

            if isinstance(status, dict):

                ready = status.get(
                    "ready",
                    False
                )

            else:

                ready = getattr(
                    status,
                    "ready",
                    False
                )

            if ready:
                break

            time.sleep(1)

        print(
            "Pinecone index created successfully."
        )

    # Return index object
    return pc.Index(index_name)


# ---------------------------------------------------------
# Pinecone Vector Store
# ---------------------------------------------------------

def get_vectorstore():
    """
    Create and cache PineconeVectorStore.
    """

    global _vectorstore

    if _vectorstore is None:

        print(
            "Initializing Pinecone vector store..."
        )

        # Ensure index exists
        index = ensure_index()

        # Get Hugging Face embeddings
        embeddings = get_embeddings()

        # Create vector store
        _vectorstore = PineconeVectorStore(
            index=index,

            embedding=embeddings,

            namespace=settings.pinecone_namespace,
        )

        print(
            "Pinecone vector store initialized."
        )

    return _vectorstore


# ---------------------------------------------------------
# Retriever
# ---------------------------------------------------------

def get_retriever():
    """
    Return Pinecone retriever.
    """

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={
            "k": settings.top_k
        }
    )


# ---------------------------------------------------------
# Add documents
# ---------------------------------------------------------

def add_documents(chunks):
    """
    Add document chunks to Pinecone.
    """

    if not chunks:
        raise ValueError(
            "No document chunks were provided."
        )

    store = get_vectorstore()

    print(
        f"Adding {len(chunks)} documents to Pinecone..."
    )

    result = store.add_documents(
        chunks
    )

    print(
        "Documents added successfully."
    )

    return result


# ---------------------------------------------------------
# Similarity Search
# ---------------------------------------------------------

def similarity_search(
    query: str,
    k: int | None = None,
):
    """
    Search Pinecone for documents similar
    to the given query.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    store = get_vectorstore()

    results = store.similarity_search(
        query=query,

        k=k or settings.top_k,
    )

    return results


# ---------------------------------------------------------
# Similarity Search With Score
# ---------------------------------------------------------

def similarity_search_with_score(
    query: str,
    k: int | None = None,
):
    """
    Search Pinecone and return similarity scores.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    store = get_vectorstore()

    results = store.similarity_search_with_score(
        query=query,

        k=k or settings.top_k,
    )

    return results