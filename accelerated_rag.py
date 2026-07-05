from matrix_bridge import MatrixBridge

if __name__ == "__main__":
    calculator = MatrixBridge()

    document_embedding = [0.12, 0.94, 0.05, 0.31]
    query_embedding    = [0.14, 0.91, 0.08, 0.28]

    similarity_score = calculator.calculate_similarity(document_embedding, query_embedding)

    print("=== MATRIXDISTANCE-RAG HARDWARE SIMD ACCELERATOR ===")
    print(f"[*] Native Unrolled Matrix Distance Vector Computation Output: {similarity_score:.4f}")
