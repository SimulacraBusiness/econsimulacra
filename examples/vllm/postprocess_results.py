from econsimulacra.log_postprocess import build_sentiment_processor

if __name__ == "__main__":
    processor = build_sentiment_processor()
    processor.process_file(
        "log_gpt-oss-120b/42.txt", "log_gpt-oss-120b_42_processed.txt"
    )
