from datasets import load_dataset
import chromadb
from tqdm.notebook import tqdm

dataset = load_dataset("sciq", split="train")

# Filter the dataset to only include questions with a support
dataset = dataset.filter(lambda x: x["support"] != "")

print("Number of questions with support: ", len(dataset))




client = chromadb.Client()


collection = client.create_collection("sciq_supports")



# Load the supporting evidence in batches of 1000
batch_size = 1000
for i in tqdm(range(0, len(dataset), batch_size), desc="Adding documents"):
    collection.add(
        ids=[
            str(i) for i in range(i, min(i + batch_size, len(dataset)))
        ],  # IDs are just strings
        documents=dataset["support"][i : i + batch_size],
        metadatas=[
            {"type": "support"} for _ in range(i, min(i + batch_size, len(dataset)))
        ],
    )

results = collection.query(
    query_texts=dataset["question"][:10],
    n_results=1)




for i, q in enumerate(dataset['question'][:10]):
    print(f"Question: {q}")
    print(f"Retrieved support: {results['documents'][i][0]}")
    print()
