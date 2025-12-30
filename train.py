import re
import torch
import random
import logging
from main import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_training_data(file_path: str) -> tuple[list[str], list[str]]:
    """Load and parse training data from file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Remove extra newlines
        text = re.sub(pattern=r"\n+", repl="\n", string=text)
        
        # Extract human and assistant interactions
        human_pattern = r'<s>Human:(.*?)</s>'
        assistant_pattern = r'<s>Assistant:(.*?)</s>'
        
        human_queries = re.findall(human_pattern, text, re.DOTALL)
        assistant_responses = re.findall(assistant_pattern, text, re.DOTALL)
        
        # Clean up the data
        human_queries = [q.strip() for q in human_queries]
        assistant_responses = [r.strip() for r in assistant_responses]
        
        # Ensure we have matching pairs
        min_pairs = min(len(human_queries), len(assistant_responses))
        if min_pairs < len(human_queries) or min_pairs < len(assistant_responses):
            logging.warning(f"数据不匹配，使用{min_pairs}对数据")
            human_queries = human_queries[:min_pairs]
            assistant_responses = assistant_responses[:min_pairs]
        
        return human_queries, assistant_responses
        
    except Exception as e:
        logging.error(f"加载训练数据失败: {e}")
        return [], []


def main():
    """Main training loop"""
    # Load training data
    human_queries, assistant_responses = load_training_data("train_sft.csv")
    
    if not human_queries or not assistant_responses:
        logging.error("没有找到训练数据，请检查train_sft.csv文件")
        return
    
    data_count = len(human_queries)
    logging.info(f"加载了{data_count}对训练数据")
    
    training_rounds = 0
    save_interval = 50  # Save model every 50 rounds
    
    try:
        while True:
            # Select a random training pair
            i = random.randint(0, data_count - 1)  # Fix: avoid index out of range
            ask = human_queries[i]
            answer = assistant_responses[i]
            
            # Skip empty queries or responses
            if not ask or not answer:
                continue
            
            # Train on this pair
            try:
                train(ask, answer)
                
                # Generate a response for evaluation
                generation(ask)
                
                training_rounds += 1
                print("*" * 100)
                
                # Save model periodically
                if training_rounds % save_interval == 0:
                    torch.save(obj=model, f="model.pth")
                    logging.info(f"模型已保存，训练轮次: {training_rounds}")
                    
            except Exception as e:
                logging.error(f"训练过程中出错: {e}")
                continue
                
    except KeyboardInterrupt:
        logging.info("训练被用户中断")
        # Save final model
        torch.save(obj=model, f="model.pth")
        logging.info(f"最终模型已保存，训练轮次: {training_rounds}")
    except Exception as e:
        logging.error(f"训练主循环出错: {e}")
        # Save model before exiting
        torch.save(obj=model, f="model.pth")


if __name__ == "__main__":
    main()