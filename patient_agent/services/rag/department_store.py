from langchain_chroma import Chroma

from patient_agent.domain.state import PatientCaseState
from patient_agent.services.rag.embeddings import get_embedding_model


PERSIST_DIR = "data/vectorstores/department_chroma"
COLLECTION_NAME = "department_triage"


def get_department_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=PERSIST_DIR,
    )


def build_department_query(state: PatientCaseState) -> str:
    facts = state.facts

    parts = [
        f"主诉：{facts.chief_complaint or ''}",
        f"症状：{'、'.join(facts.symptoms)}",
        f"持续时间：{facts.duration or ''}",
        f"严重程度：{facts.severity or ''}",
        f"既往史：{'、'.join(facts.medical_history)}",
        f"当前用药：{'、'.join(facts.medications)}",
        f"过敏史：{'、'.join(facts.allergies)}",
        f"红旗风险：{'、'.join(state.red_flags)}",
        f"图片分析：{'、'.join(state.image_findings)}",
    ]

    return "\n".join(part for part in parts if part.strip())


def search_department_candidates(
    state: PatientCaseState,
    k: int = 5,
) -> list[dict]:
    vector_store = get_department_vector_store()
    query = build_department_query(state)

    results = vector_store.similarity_search_with_score(query, k=k)

    candidates: list[dict] = []

    for doc, score in results:
        candidates.append(
            {
                "department": doc.metadata.get("department"),
                "priority": doc.metadata.get("priority"),
                "source_id": doc.metadata.get("id"),
                "score": float(score),
                "content": doc.page_content,
                "metadata": dict(doc.metadata),
            }
        )

    return candidates