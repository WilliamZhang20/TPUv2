import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchao.float8 import (
    convert_to_float8_training,
    Float8LinearConfig,
)
from test_tpu import reset_dut
import cocotb
from cocotb.clock import Clock

BATCH_SIZE = 32

class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def compute_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def get_quantized_model():
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = FCNet()

    # --- FP8 QAT SETUP ---
    print("Converting model to FP8 training mode...")
    
    # Configure FP8 quantization using a recipe
    # "tensorwise" is the fastest and most common recipe
    config = Float8LinearConfig.from_recipe_name("tensorwise")
    
    # Convert model to FP8 training - this wraps linear layers
    model = convert_to_float8_training(model, config=config)
    print("Model converted to FP8 training mode")

    # --- QAT TRAINING LOOP ---
    print("FP8 QAT Training begins")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(3):
        model.train()
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass - linear layers now use FP8 with dynamic scales
            out = model(images)
            loss = criterion(out, labels)
            
            # Backward pass - gradients computed in FP8
            loss.backward()
            optimizer.step()

        # Accuracy after epoch
        model.eval()
        acc = compute_accuracy(model, train_loader)
        print(f"Epoch {epoch+1} training accuracy: {acc:.4f}")

    print("FP8 QAT Training complete")
    
    # After training, model is ready for FP8 inference
    model.eval()
    
    return model

# 3. Hardware Test Function 
@cocotb.test()
async def tpu_torch_test(dut):
    # build model with FP8 QAT
    model = get_quantized_model()
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # compile it with backend
    from torch_backend import make_backend
    backend = make_backend(dut) 
    
    # torch.compile compiles the FP8 quantized module
    compiled_model = torch.compile(model, backend=backend)

    # Load a few samples
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    test_correct = 0
    test_total = 0

    for i, (image, label) in enumerate(test_loader):
        test_total += 1
        if i >= 5:
            break
        # RUN INFERENCE IN SEPARATE THREAD
        import concurrent.futures
        from cocotb.triggers import Timer

        def run_inference():
            with torch.no_grad():
                return compiled_model(image)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_inference)

            # POLL SIMULATOR WHILE WAITING
            while not future.done():
                await Timer(10, units='ns')  # Keep cocotb alive

            dut_out = future.result()

            print("Predicted Label:", torch.argmax(dut_out, dim=1).item())
            print("Actual label:", label.item())
            if (torch.argmax(dut_out, dim=1).item() == label.item()):
                test_correct += 1

    accuracy = test_correct / test_total
    assert accuracy >= 0.7, f"Test accuracy too low: {accuracy:.4f}"

    print("TEST PASSED")