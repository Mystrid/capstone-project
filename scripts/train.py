import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from egasn.models.egasn import EGASN
from egasn.data.cityscapes import CityscapesVideoDataset
from egasn.utils.metrics import Evaluator
from egasn.losses.evidential import EvidentialLoss, TemporalConsistencyLoss
from egasn.graph.epipolar import EpipolarGraphConstructor

def parse_args():
    parser = argparse.ArgumentParser(description="Train EGASN on Cityscapes")
    parser.add_argument("--config", type=str, default="configs/egasn_cityscapes.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

def validate(model, val_loader, graph_constructor, device):
    """
    Validation loop.
    
    Args:
        model: EGASN model.
        val_loader: DataLoader for validation data.
        graph_constructor: EpipolarGraphConstructor instance.
        device: Torch device.
        
    Returns:
        float: Mean IoU.
    """
    model.eval()
    evaluator = Evaluator(num_classes=19)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            frame_prev_np = batch['frame_prev_np']
            frame_curr_np = batch['frame_curr_np']
            frame_curr = batch['frame_curr'].to(device)
            label = batch['label'].to(device)
            
            # Simulated previous features for validation (in real video, keep state)
            b, c, h, w = frame_curr.shape
            x_prev_features = torch.zeros((b, 256, h // 4, w // 4), device=device) # Dummy
            
            flow_fields = []
            for i in range(b):
                flow, _ = graph_constructor.construct(frame_prev_np[i], frame_curr_np[i])
                flow_fields.append(flow)
            flow_batch = torch.stack(flow_fields).to(device)
            
            output = model(frame_curr, x_prev_features, flow_batch)
            preds = output['probs'].argmax(dim=1)
            evaluator.add_batch(label, preds)
            
    miou = evaluator.Mean_Intersection_over_Union()
    return miou

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(args.device)
    
    # Datasets
    train_dataset = CityscapesVideoDataset(split='train', transforms=None)
    val_dataset = CityscapesVideoDataset(split='val', transforms=None)
    
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 4), shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Model
    model = EGASN(num_classes=19).to(device)
    
    # Optimizer & Scheduler
    epochs = config.get('epochs', 100)
    optimizer = optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=config.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Losses
    evidential_loss_fn = EvidentialLoss(num_classes=19, annealing_epochs=epochs // 2)
    tc_loss_fn = TemporalConsistencyLoss()
    
    graph_constructor = EpipolarGraphConstructor()
    
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0.0)
        
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir="runs/egasn")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        evidential_loss_fn.set_epoch(epoch)
        
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            frame_prev_np = batch['frame_prev_np']
            frame_curr_np = batch['frame_curr_np']
            frame_curr = batch['frame_curr'].to(device)
            label = batch['label'].to(device)
            
            b, c, h, w = frame_curr.shape
            # Dummy state for training batch
            x_prev_features = torch.zeros((b, 256, h // 4, w // 4), device=device)
            prev_probs_warped = torch.ones((b, 19, h, w), device=device) / 19.0
            valid_mask = torch.ones((b, 1, h, w), device=device)
            
            flow_fields = []
            for i in range(b):
                flow, _ = graph_constructor.construct(frame_prev_np[i], frame_curr_np[i])
                flow_fields.append(flow)
            flow_batch = torch.stack(flow_fields).to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output = model(frame_curr, x_prev_features, flow_batch)
                
                tc_loss_val = tc_loss_fn(output['probs'], prev_probs_warped, valid_mask)
                loss = evidential_loss_fn(output, label, tc_loss_val)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_miou = validate(model, val_loader, graph_constructor, device)
            writer.add_scalar('mIoU/val', val_miou, epoch)
            
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_miou': max(best_miou, val_miou)
            }
            torch.save(state, "latest_model.pth")
            
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(state, "best_model.pth")
                
    writer.close()

if __name__ == "__main__":
    main()
