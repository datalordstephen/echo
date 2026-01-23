import soundata
import os
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("echo_download")

def download_urbansound8k(data_home='urbansound8k'):
    """
    Downloads and validates the UrbanSound8K dataset.
    """
    logger.info(f"Initializing UrbanSound8K in '{data_home}'...")
    dataset = soundata.initialize('urbansound8k', data_home=data_home)
    
    logger.info("Downloading dataset (this may take a while)...")
    dataset.download()  
    
    logger.info("Validating dataset...")
    dataset.validate() 
    
    logger.info("✅ Dataset downloaded and validated successfully!")
    
    # Example usage
    example_clip = dataset.choice_clip()
    print("\n--- Example Clip Metadata ---")
    print(example_clip)

if __name__ == "__main__":
    download_urbansound8k()
