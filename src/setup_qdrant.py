import os
import json
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env.local'))

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
qdrant_client = QdrantClient(path="./qdrant_db")

COLLECTION_NAME = "dental_knowledge"
DATA_PATH = "./data"


def create_collection():
    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print("✅ Collection created")


def get_embedding(text):
    return openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    ).data[0].embedding


def extract_metadata(filename):
    name = filename.lower()
    meta = {}

    meta["type"] = "faq" if "faq" in name else "object"

    if "clinic" in name:
        meta["category"] = "clinic_info"
    elif "insurance" in name:
        meta["category"] = "insurance"
    elif "polic" in name:
        meta["category"] = "policy"
    elif "doctor" in name or "profile" in name:
        meta["category"] = "doctor"
    elif any(x in name for x in ["implant", "crown", "denture", "surgery", "cosmetic", "treatment"]):
        meta["category"] = "treatment"
    else:
        meta["category"] = "general"

    for t in ["implant", "crown", "denture", "surgery", "cosmetic"]:
        if t in name:
            meta["treatment"] = t

    meta["source"] = filename
    return meta


def chunk_json(data, meta):
    chunks = []

    if meta["type"] == "faq":
        for item in data:
            if "question" in item and "answer" in item:
                chunks.append(f"Q: {item['question']} A: {item['answer']}")
    else:
        if isinstance(data, dict):
            if "description" in data:
                chunks.append(data["description"])

            if "locations" in data:
                for loc in data["locations"]:
                    addr = loc.get("address", {})
                    chunks.append(
                        f"{loc.get('clinic_name', '')} located at {addr.get('street', '')} {addr.get('city', '')}"
                    )

            if "languages" in data:
                doc = ", ".join(data["languages"].get("doctor", []))
                staff = ", ".join(data["languages"].get("staff", []))
                chunks.append(f"Doctor speaks {doc}. Staff speaks {staff}")

            if "main_services" in data:
                chunks.append("Services include: " + ", ".join(data["main_services"]))

            if "notes" in data:
                chunks.append(data["notes"])

            if not chunks:
                chunks.append(json.dumps(data))

    return chunks


def ingest_json():
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".json")]
    points = []

    for file in files:
        path = os.path.join(DATA_PATH, file)

        if os.path.getsize(path) == 0:
            print(f"⚠️  Skipping empty file: {file}")
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️  Skipping invalid JSON: {file} ({e})")
            continue

        meta = extract_metadata(file)
        chunks = chunk_json(data, meta)

        print(f"📂 {file} → {len(chunks)} chunks")

        for chunk in chunks:
            embedding = get_embedding(chunk)
            payload = {
                "content": chunk,
                "category": meta.get("category"),
                "type": meta.get("type"),
                "treatment": meta.get("treatment", ""),
                "source": meta.get("source")
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
            )

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Uploaded {len(points)} vectors")


if __name__ == "__main__":
    create_collection()
    ingest_json()
    print("🔥 JSON RAG READY")
