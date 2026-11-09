"""
Phase 4 Independent Forensic Verification and Metric Reconstruction Script.
Independently verifies and reconstructs all validation and held-out test metrics,
computes confusion matrices, macro-F1, balanced accuracy, per-class metrics,
and saves per-sample test predictions.
"""

import sys
import os
import json
import hashlib
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

# Ensure project root is in sys.path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from research.data.pathmnist_loader import get_test_data, get_validation_data
from research.models.compact_cnn import CompactPathCNN, preprocess_batch


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, num_classes: int = 9) -> Dict[str, Any]:
    """
    Computes confusion matrix, macro-F1, balanced accuracy, and per-class precision, recall, F1, and support.
    """
    # 1. Confusion Matrix: rows = true, cols = pred
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    per_class = {}
    recalls = []
    f1s = []
    
    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(np.sum(cm[c, :]) - tp)
        fp = int(np.sum(cm[:, c]) - tp)
        support = int(np.sum(cm[c, :]))
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        per_class[str(c)] = {
            "precision": float(round(precision, 5)),
            "recall": float(round(recall, 5)),
            "f1_score": float(round(f1, 5)),
            "support": support
        }
        recalls.append(recall)
        f1s.append(f1)
        
    macro_f1 = float(np.mean(f1s))
    balanced_acc = float(np.mean(recalls))
    overall_acc = float(np.sum(y_true == y_pred) / len(y_true))
    
    return {
        "overall_accuracy": float(round(overall_acc, 5)),
        "balanced_accuracy": float(round(balanced_acc, 5)),
        "macro_f1": float(round(macro_f1, 5)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class
    }


@torch.no_grad()
def evaluate_checkpoint_on_test(ckpt_path: str, test_images: np.ndarray, test_labels: np.ndarray, pred_out_path: str = None) -> Dict[str, Any]:
    """
    Loads a checkpoint with strict state-dict verification, evaluates on official held-out test split,
    computes detailed metrics, and optionally exports per-sample predictions.
    """
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    ckpt_hash = compute_file_sha256(ckpt_path)
    
    model = CompactPathCNN()
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    total_samples = len(test_labels)
    all_preds = []
    all_probs = []
    all_targets = test_labels.squeeze()
    total_loss = 0.0
    batch_size = 128
    num_batches = int(np.ceil(total_samples / batch_size))
    
    for b in range(num_batches):
        idx_slice = slice(b * batch_size, min((b + 1) * batch_size, total_samples))
        b_imgs = test_images[idx_slice]
        b_lbls = all_targets[idx_slice]
        
        x, y = preprocess_batch(b_imgs, b_lbls)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        
    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_probs = np.array(all_probs, dtype=float)
    
    metrics = compute_classification_metrics(y_true, y_pred, y_probs)
    metrics["test_loss"] = float(round(total_loss / total_samples, 5))
    metrics["checkpoint_path"] = os.path.abspath(ckpt_path)
    metrics["checkpoint_sha256"] = ckpt_hash
    metrics["total_test_samples"] = total_samples
    
    # Save per-sample predictions if requested
    if pred_out_path is not None:
        os.makedirs(os.path.dirname(pred_out_path), exist_ok=True)
        with open(pred_out_path, "w", encoding="utf-8") as f:
            for idx in range(total_samples):
                t_cls = int(y_true[idx])
                p_cls = int(y_pred[idx])
                conf = float(y_probs[idx, p_cls])
                rec = {
                    "sample_index": idx,
                    "true_class": t_cls,
                    "predicted_class": p_cls,
                    "confidence": float(round(conf, 5)),
                    "correct": bool(t_cls == p_cls)
                }
                f.write(json.dumps(rec) + "\n")
                
    return metrics


def audit_experiment_jsonl(jsonl_path: str) -> Dict[str, Any]:
    """
    Parses and reconstructs audit metrics directly from raw JSONL round records.
    """
    assert os.path.exists(jsonl_path), f"JSONL missing: {jsonl_path}"
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    total_rounds = len(records)
    best_acc = -1.0
    best_loss = 1e9
    best_round = -1
    
    for r in records:
        acc = r["val/global_accuracy"]
        loss = r["val/global_loss"]
        rnd = r["round"]
        if acc > best_acc or (acc == best_acc and loss < best_loss):
            best_acc = acc
            best_loss = loss
            best_round = rnd
            
    round_durations = [r["round_duration_seconds"] for r in records]
    rss_measurements = [r["process_rss_mb"] for r in records]
    clean_noisy_alignments = [r["diagnostics/mean_clean_noisy_cosine"] for r in records]
    clean_clean_alignments = [r["diagnostics/mean_clean_clean_cosine"] for r in records]
    
    update_norms = {}
    for c in range(1, 6):
        update_norms[f"client_{c}"] = float(np.mean([r[f"diagnostics/client_{c}_update_l2_norm"] for r in records]))
        
    return {
        "file_path": os.path.abspath(jsonl_path),
        "total_rounds_logged": total_rounds,
        "best_round": best_round,
        "best_val_accuracy": float(round(best_acc, 5)),
        "best_val_loss": float(round(best_loss, 5)),
        "mean_round_duration_sec": float(round(np.mean(round_durations), 2)),
        "total_duration_minutes": float(round(np.sum(round_durations) / 60.0, 2)),
        "peak_rss_mb": float(round(max(rss_measurements), 2)),
        "mean_clean_noisy_alignment": float(round(np.mean(clean_noisy_alignments), 5)),
        "mean_clean_clean_consensus": float(round(np.mean(clean_clean_alignments), 5)),
        "mean_update_norms": update_norms
    }


def run_full_forensic_audit():
    print("======================================================================")
    print("      PHASE 4 FORENSIC AUDIT & INDEPENDENT METRIC RECONSTRUCTION      ")
    print("======================================================================")
    
    results_dir = os.path.join(PROJ_ROOT, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Audit Checkpoints on Held-Out Test Split
    print("\n1. Evaluating saved best checkpoints on official held-out test split (7,180 samples)...")
    test_images, test_labels = get_test_data()
    
    ckpt_dir = os.path.join(PROJ_ROOT, "experiments", "checkpoints")
    fedavg_ckpt = os.path.join(ckpt_dir, "fedavg_alpha0.3_noise40_seed42_best.pt")
    fedprox_ckpt = os.path.join(ckpt_dir, "fedprox_mu0.01_alpha0.3_noise40_seed42_best.pt")
    
    fedavg_pred_file = os.path.join(results_dir, "fedavg_test_predictions.jsonl")
    fedprox_pred_file = os.path.join(results_dir, "fedprox_test_predictions.jsonl")
    
    fedavg_test_audit = evaluate_checkpoint_on_test(fedavg_ckpt, test_images, test_labels, fedavg_pred_file)
    fedprox_test_audit = evaluate_checkpoint_on_test(fedprox_ckpt, test_images, test_labels, fedprox_pred_file)
    
    print(f"   FedAvg Test Acc: {fedavg_test_audit['overall_accuracy'] * 100:.2f}%, Macro-F1: {fedavg_test_audit['macro_f1']:.4f}, Balanced Acc: {fedavg_test_audit['balanced_accuracy'] * 100:.2f}%, Loss: {fedavg_test_audit['test_loss']:.4f}")
    print(f"   FedProx Test Acc: {fedprox_test_audit['overall_accuracy'] * 100:.2f}%, Macro-F1: {fedprox_test_audit['macro_f1']:.4f}, Balanced Acc: {fedprox_test_audit['balanced_accuracy'] * 100:.2f}%, Loss: {fedprox_test_audit['test_loss']:.4f}")
    
    # Save Confusion Matrices
    with open(os.path.join(results_dir, "fedavg_confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(fedavg_test_audit["confusion_matrix"], f, indent=2)
    with open(os.path.join(results_dir, "fedprox_confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(fedprox_test_audit["confusion_matrix"], f, indent=2)
        
    # 2. Audit Raw Logs
    print("\n2. Auditing raw JSONL telemetry logs...")
    log_dir = os.path.join(PROJ_ROOT, "experiments", "logs")
    fedavg_log_audit = audit_experiment_jsonl(os.path.join(log_dir, "fedavg_metrics.jsonl"))
    fedprox_log_audit = audit_experiment_jsonl(os.path.join(log_dir, "fedprox_metrics.jsonl"))
    
    print(f"   FedAvg: {fedavg_log_audit['total_rounds_logged']} rounds, Best Round: {fedavg_log_audit['best_round']} (Acc: {fedavg_log_audit['best_val_accuracy'] * 100:.2f}%, Loss: {fedavg_log_audit['best_val_loss']:.4f})")
    print(f"   FedProx: {fedprox_log_audit['total_rounds_logged']} rounds, Best Round: {fedprox_log_audit['best_round']} (Acc: {fedprox_log_audit['best_val_accuracy'] * 100:.2f}%, Loss: {fedprox_log_audit['best_val_loss']:.4f})")
    
    audit_summary = {
        "forensic_audit_date": "2026-11-07",
        "official_held_out_test_evaluation": {
            "fedavg": fedavg_test_audit,
            "fedprox": fedprox_test_audit
        },
        "raw_jsonl_log_reconstruction": {
            "fedavg": fedavg_log_audit,
            "fedprox": fedprox_log_audit
        }
    }
    
    audit_path = os.path.join(results_dir, "recomputed_metrics_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
        
    print(f"\nAudit complete! Full audit records written to {audit_path}")


if __name__ == "__main__":
    run_full_forensic_audit()
