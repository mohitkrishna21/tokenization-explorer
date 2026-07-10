from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

# ── Load models ──────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def explore_tokenization(text):
    print("\n" + "="*60)
    print(f"INPUT TEXT: {text}")
    print("="*60)

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)

    print(f"\n📌 TOKENS ({len(tokens)} total):")
    print(tokens)

    print(f"\n🔢 TOKEN IDs:")
    print(token_ids)

    print(f"\n📖 TOKEN → ID MAPPING:")
    for token, tid in zip(tokens, token_ids[1:-1]):
        print(f"   '{token}' → {tid}")

def explore_embeddings(text):
    embedding = embedding_model.encode(text)
    print(f"\n🧠 EMBEDDING for: '{text}'")
    print(f"   Vector size: {embedding.shape[0]} dimensions")
    print(f"   First 5 values: {embedding[:5].round(4)}")
    return embedding

def semantic_similarity(text1, text2):
    emb1 = embedding_model.encode(text1)
    emb2 = embedding_model.encode(text2)

    similarity = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

    print(f"\n🔍 SEMANTIC SIMILARITY")
    print(f"   Text 1: '{text1}'")
    print(f"   Text 2: '{text2}'")
    print(f"   Similarity Score: {similarity:.4f}")

    if similarity > 0.8:
        print("   Verdict: 🟢 Very similar")
    elif similarity > 0.5:
        print("   Verdict: 🟡 Somewhat similar")
    else:
        print("   Verdict: 🔴 Not similar")

# ── Main Menu ─────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        print("\n" + "="*60)
        print("🤖 TOKENIZATION & EMBEDDING EXPLORER")
        print("="*60)
        print("1. Explore Tokenization")
        print("2. Explore Embeddings")
        print("3. Check Semantic Similarity")
        print("4. Exit")
        print("="*60)

        choice = input("Choose an option (1-4): ").strip()

        if choice == "1":
            text = input("\nEnter text to tokenize: ").strip()
            explore_tokenization(text)

        elif choice == "2":
            text = input("\nEnter text to get embedding: ").strip()
            explore_embeddings(text)

        elif choice == "3":
            text1 = input("\nEnter first sentence: ").strip()
            text2 = input("Enter second sentence: ").strip()
            semantic_similarity(text1, text2)

        elif choice == "4":
            print("\nBye! 👋")
            break

        else:
            print("\n❌ Invalid option. Please choose 1-4.")