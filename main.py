import random
import torch
from model import MainModel, CONFIG
from record import record_loss

# Special tokens
START_TOKEN = 1
END_TOKEN = CONFIG['dict_size'] - 1

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoding/Decoding functions
# Note: This is a simple ASCII-based encoding. For better performance, consider using a proper tokenizer.
def encode(text: str) -> torch.Tensor:
    """Encode text to tensor of indices"""
    tensor = []
    for letter in text:
        try:
            idx = ord(letter)
            # Ensure index is within the dictionary size
            if 0 < idx < CONFIG['dict_size'] - 1:
                tensor.append(idx)
            else:
                tensor.append(0)  # Unknown token
        except:
            tensor.append(0)  # Unknown token
    return torch.tensor(tensor, dtype=torch.long)

def decode(indices: torch.Tensor) -> str:
    """Decode tensor of indices to text"""
    text = []
    for idx in indices:
        try:
            if idx != START_TOKEN and idx != END_TOKEN:
                text.append(chr(idx))
        except:
            continue
    return ''.join(text)

def one_hot_encode(idx: int) -> torch.Tensor:
    """Create one-hot encoding for a token index"""
    tensor = torch.zeros(CONFIG['dict_size'], dtype=torch.float32)
    if 0 <= idx < CONFIG['dict_size']:
        tensor[idx] = 1.0
    return tensor

# Model initialization
print(f"使用设备: {device}")
try:
    model = torch.load(f="model.pth", map_location=device, weights_only=False)
    print("载入模型成功")
    # Verify model structure compatibility
    if not hasattr(model, 'transformers') or not hasattr(model.transformers[0], 'rms_norm1'):
        print("检测到旧模型结构，创建新模型")
        model = MainModel().to(device)
except Exception as e:
    print(f"载入模型失败: {e}")
    model = MainModel().to(device)
    print("创建新模型")

# Loss function and optimizer
loss_func = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

def train(ask: str, answer: str) -> None:
    """Train the model on a single (ask, answer) pair"""
    print(f"\n---训练问题:\n{ask}")
    print("\n---训练回答:")
    
    # Prepare input tensors
    ask_tensor = encode(ask).to(device)
    answer_tensor = encode(answer).to(device)
    
    # Create training sequence: [ask tokens] + [start token] + [answer tokens] + [end token]
    train_tensor = torch.cat([
        ask_tensor,
        torch.tensor([START_TOKEN], device=device),
        answer_tensor,
        torch.tensor([END_TOKEN], device=device)
    ])
    
    # Training loop
    model.train()
    for i in range(len(ask_tensor) + 1, len(train_tensor)):
        # Get context up to current position
        context = train_tensor[:i]
        
        # Current token to predict is train_tensor[i]
        target_idx = train_tensor[i]
        target = one_hot_encode(int(target_idx)).to(device)
        
        # Autoregressive input is the last token in context
        autoregressive_input = context[-1].unsqueeze(0)
        
        # Forward pass
        output = model(autoregressive_input, context)
        loss = loss_func(output, target)
        
        # Record loss
        record_loss(loss.item())
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Print the generated token for monitoring
        print(chr(int(target_idx)), end="", flush=True)

    print("", flush=True)  # New line after generation

def generation(text: str) -> str:
    """Generate a response to the input text"""
    model.eval()
    output_text = ""
    
    # Prepare initial prompt
    prompt = torch.cat([
        encode(text).to(device),
        torch.tensor([START_TOKEN], device=device)
    ])
    
    print("\n---生成回复:")
    
    with torch.no_grad():
        for _ in range(CONFIG['max_length']):
            autoregressive_input = prompt[-1].unsqueeze(0).to(device)
            output = model(autoregressive_input, prompt)
            
            # Sample from the output distribution
            probs = torch.softmax(output / CONFIG['temperature'], dim=-1)
            index = int(torch.multinomial(probs, 1))
            
            # Check if we've reached the end token
            if index == END_TOKEN:
                break
            
            # Append to output text if not special token
            if index != START_TOKEN:
                try:
                    char = chr(index)
                    print(char, end="", flush=True)
                    output_text += char
                except:
                    continue
            
            # Update prompt with the new token
            prompt = torch.cat([prompt, torch.tensor([index], device=device)])
    
    print("", flush=True)  # New line after generation
    return output_text