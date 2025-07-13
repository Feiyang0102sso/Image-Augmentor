import logging

# General logging configuration
def setup_general_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Demo logging configuration
def setup_demo_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("demo_log.txt"),
            logging.StreamHandler()
        ]
    )