import re
import random
import logging
from main import generation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_test_data(file_path: str) -> list[str]:
    """Load test data from file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Extract human queries
        pattern = r'<s>Human:(.*?)</s>'
        queries = re.findall(pattern, text, re.DOTALL)
        
        # Clean up the data
        queries = [q.strip() for q in queries]
        
        return queries
        
    except Exception as e:
        logging.error(f"加载测试数据失败: {e}")
        return []


def main():
    """Main test loop"""
    # Load test data
    test_queries = load_test_data("train_sft.csv")
    
    if not test_queries:
        logging.error("没有找到测试数据，请检查train_sft.csv文件")
        return
    
    logging.info(f"加载了{len(test_queries)}条测试数据")
    
    try:
        while True:
            # Select a random test query
            i = random.randint(0, len(test_queries) - 1)
            ask = test_queries[i]
            
            if not ask:
                continue
            
            print(f"\n=== 测试查询: ===")
            print(ask)
            
            # Generate a response
            generation(ask)
            
            print("*" * 100)
            
            # Ask for user input to continue
            input("按Enter键继续，按Ctrl+C退出...")
            
    except KeyboardInterrupt:
        logging.info("测试结束")
    except Exception as e:
        logging.error(f"测试过程中出错: {e}")


if __name__ == "__main__":
    main()