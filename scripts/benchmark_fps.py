import argparse
import time
import torch
import numpy as np

from egasn.models.egasn import EGASN
from egasn.graph.epipolar import EpipolarGraphConstructor

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark EGASN FPS")
    parser.add_argument("--config", type=str, default="configs/egasn_cityscapes.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations for benchmarking")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    model = EGASN(num_classes=19).to(device)
    model.eval()
    
    graph_constructor = EpipolarGraphConstructor(output_size=(768 // 32, 768 // 32))
    
    h, w = 768, 768
    dummy_input = torch.randn(1, 3, h, w, device=device)
    dummy_prev_feat = torch.randn(1, 256, h // 32, w // 32, device=device)
    
    dummy_img_prev = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    dummy_img_curr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            flow, _ = graph_constructor.construct(dummy_img_prev, dummy_img_curr)
            flow_tensor = flow.unsqueeze(0).to(device)
            _ = model(dummy_input, dummy_prev_feat, flow_tensor)
            
    print(f"Running benchmark for {args.num_iterations} iterations...")
    
    times = {'graph': [], 'backbone': [], 'attention': [], 'decoder': [], 'total': []}
    
    with torch.no_grad():
        for _ in range(args.num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            
            # Graph
            flow, _ = graph_constructor.construct(dummy_img_prev, dummy_img_curr)
            flow_tensor = flow.unsqueeze(0).to(device)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            
            # End-to-end
            _ = model(dummy_input, dummy_prev_feat, flow_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.time()
            
            times['graph'].append(t1 - t0)
            times['total'].append(t2 - t0)
            # Detailed breakdown requires hooks, approximating here:
            # For exact module timing, we'd add timers inside EGASN's forward
            
    avg_total = np.mean(times['total'])
    avg_graph = np.mean(times['graph'])
    avg_net = avg_total - avg_graph
    
    print("-" * 40)
    print("BENCHMARK RESULTS")
    print("-" * 40)
    print(f"OpenCV Graph: {avg_graph * 1000:.2f} ms")
    print(f"Network Pass: {avg_net * 1000:.2f} ms")
    print(f"Total Time:   {avg_total * 1000:.2f} ms")
    print(f"Total FPS:    {1.0 / avg_total:.2f}")
    print("-" * 40)

if __name__ == "__main__":
    main()
