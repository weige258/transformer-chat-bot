import logging
from main import generation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """Main chat loop"""
    print("=== Transformer 聊天机器人 ===")
    print("输入 'quit' 或 'exit' 退出聊天")
    print("输入 'clear' 清空屏幕")
    print("=" * 50)
    
    try:
        while True:
            try:
                user_input = input("\n你: ")
                
                # Handle commands
                if user_input.lower() in ["quit", "exit"]:
                    print("再见!")
                    break
                elif user_input.lower() == "clear":
                    # Clear screen
                    import os
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("=== Transformer 聊天机器人 ===")
                    print("输入 'quit' 或 'exit' 退出聊天")
                    print("输入 'clear' 清空屏幕")
                    print("=" * 50)
                    continue
                
                # Skip empty input
                if not user_input.strip():
                    continue
                
                # Generate response
                response = generation(user_input)
                
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except Exception as e:
                logging.error(f"聊天过程中出错: {e}")
                print("抱歉，发生了一些错误，请重试。")
    
    except Exception as e:
        logging.error(f"聊天程序出错: {e}")


if __name__ == "__main__":
    main()