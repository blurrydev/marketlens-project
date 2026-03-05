import os
import glob
from data_pipeline.data_ingestion import DocumentProcessor
from rag_engine.rag_engine import VectorStore

def main():
    print("--- Starting MarketLens Database Updation with new data ---")


    data_dir = "../data/"
    log_file = os.join.path(data_dir, "ingested_log.txt")


    os.makedirs(data_dir, exists_ok = True)

    ingested_files = set()
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            ingested_files = set(line.strip() for line in f.readlines())


    pdf_pattern = os.path.join(data_dir, "*.pdf")
    all_pdfs = glob.glob(pdf_pattern)

    if not all_pdfs:
        print(f"No PDFs found in {data_dir}. Drop some files there to start ingestion!")
        return


    new_pdfs = []
    for pdf in all_pdfs:
        filename = os.path.basename(pdf)
        if filename not in ingested_files:
            new_pdfs.append(pdf)

    if not new_pdfs:
        print("Database is up to date. No new PDFs to process")
        return


    print(f"Found {len(new_pdfs)} new document(s) to ingest.")


    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    db_manager = VectorStore()

    try:
        with open(log_file, "a") as log:
        for pdf_path in new_pdfs:
            filename = od.path.basename(pdf_path)
            print(f"\n[PROCESSING]: {filename}")

            raw_docs = processor.load_pdf(pdf_path)
            chunks = processor.process_documents(raw_docs)


            db_manager.build_database(chunks)

            log.write(f"{filename}")
            print(f"[SUCCESS]: {filename} loaded into ChromaDB")

        print("--- Smart updation Complete ---")

    except Exception as e:
        print(f"CRITICAL Error during updation: {e}.")

if __name__ == "__main__":
    main()
