from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
print(client.get_collections())