import json
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_chroma import Chroma

from patient_agent.services.rag.embeddings import get_embedding_model


SOURCE_PATH = Path("data/triage_knowledge/departments.jsonl")
PERSIST_DIR = "data/vectorstores/department_chroma"
COLLECTION_NAME = "department_triage"


def item_to_document(item: dict) -> Document:
    content = f"""
科室：{item["department"]}
优先级：{item.get("priority", "")}

适合主诉：
{"、".join(item.get("chief_complaints", []))}

相关症状：
{"、".join(item.get("symptoms", []))}

红旗风险：
{"、".join(item.get("red_flags", []))}

适合场景：
{item.get("suitable_for", "")}

不适合场景：
{item.get("not_suitable_for", "")}

推荐理由：
{item.get("department_reason", "")}
""".strip()

    metadata = {
        "id": item["id"],
        "department": item["department"],
        "priority": item.get("priority", ""),
        "red_flags": ",".join(item.get("red_flags", [])),
        "source": "department_triage_knowledge",
    }

    return Document(page_content=content, metadata=metadata)


def load_documents() -> list[Document]:
    documents: list[Document] = []

    for line in SOURCE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)
        documents.append(item_to_document(item))

    return documents


def main():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE_PATH}")

    embeddings = get_embedding_model()
    documents = load_documents()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # Rebuild simple version: delete known ids by resetting directory manually if needed.
    ids = [str(uuid4()) for _ in documents]
    vector_store.add_documents(documents=documents, ids=ids)

    print(f"Indexed {len(documents)} department documents")
    print(f"Persist directory: {PERSIST_DIR}")


if __name__ == "__main__":
    main()