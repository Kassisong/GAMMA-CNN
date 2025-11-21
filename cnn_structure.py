import os
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Union, Sequence, Dict, Any, Tuple, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

# -------------------------
# Layer descriptors (not nn.Module)
# -------------------------
class Layer(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        ...

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SkipLayer(Layer):
    GROUP_NUMBER = 1

    def __init__(self,
                 feature_size1: int,
                 feature_size2: int,
                 kernel: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 convolution: str = 'same'):
        self.convolution = convolution
        self.stride = stride
        self.kernel = kernel
        self.feature_size2 = feature_size2
        self.feature_size1 = feature_size1

    def __repr__(self) -> str:
        return f"{self.feature_size1}-{self.feature_size2}"

class _MLPBlock(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=128, output_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),   #  forward  in_features
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU()
        )

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class PoolingLayer(Layer):
    def __init__(self, pooling_type: str, kernel: Tuple[int, int] = (2, 2), stride: Tuple[int, int] = (2, 2)):
        assert pooling_type in ("max", "mean")
        self.pooling_type = pooling_type
        self.kernel = kernel
        self.stride = stride

    def __repr__(self) -> str:
        return self.pooling_type

class _SkipBlock(nn.Module):
    def __init__(self, in_ch, f1, f2, kernel=(3,3), stride=(1,1), padding_mode="same", p_drop=0.2):
        super().__init__()
        padding = "same" if padding_mode == "same" else 0
        self.conv1 = nn.Conv2d(in_ch, f1, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.drop = nn.Dropout2d(p=p_drop)

        self.proj = nn.Conv2d(in_ch, f2, kernel_size=1, stride=stride, bias=False)
        self.final_act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.proj(x)
        out = self.drop(self.act1(self.bn1(self.conv1(x))))
        out = self.drop(self.bn2(self.conv2(out)))
        out = out + identity
        out = self.final_act(out)
        return out

class _Pooling(nn.Module):
    def __init__(self, pooling_type: str, kernel: Tuple[int, int], stride: Tuple[int, int]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel, stride=stride)

    def forward(self, x):
        return self.pool(x)
# -------------------------
# Fusion utilities
# -------------------------

class _AttentionFuse(nn.Module):
    """Learnable attention fusion with automatic spatial alignment."""
    def __init__(self, num_streams: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(num_streams))

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        # ----  ----
        max_h = max(x.shape[2] if x.ndim > 2 else 1 for x in xs)
        max_w = max(x.shape[3] if x.ndim > 3 else 1 for x in xs)
        xs_resized = [_resize_to_match(x, (max_h, max_w)) for x in xs]

        # ---- soft attention  ----
        w = torch.softmax(self.alpha, dim=0)
        out = 0
        for i, x in enumerate(xs_resized):
            out = out + w[i] * x
        return out

def _resize_to_match(t: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """ resize  target_hw=(H,W)， batch  channel """
    if t.ndim == 2:  # (N, L)
        t = t.unsqueeze(1).unsqueeze(2)  # -> (N,1,1,L)
    if t.ndim == 3:  # (N,C,H)
        t = t.unsqueeze(1)  # -> (N,1,C,H)
    if t.ndim == 4:
        _, _, h, w = t.shape
        th, tw = target_hw
        if (h, w) != (th, tw):
            t = F.interpolate(t, size=(th, tw), mode='bilinear', align_corners=False)
    return t

def _fuse_add(xs: List[torch.Tensor]) -> torch.Tensor:
    """ add """
    # H,W
    max_h = max(x.shape[2] if x.ndim > 2 else 1 for x in xs)
    max_w = max(x.shape[3] if x.ndim > 3 else 1 for x in xs)
    xs_resized = [_resize_to_match(x, (max_h, max_w)) for x in xs]

    y = xs_resized[0]
    for x in xs_resized[1:]:
        y = y + x
    return y

def _fuse_concat(xs: List[torch.Tensor], dim: int = 1) -> torch.Tensor:
    """ concat """
    max_h = max(x.shape[2] if x.ndim > 2 else 1 for x in xs)
    max_w = max(x.shape[3] if x.ndim > 3 else 1 for x in xs)
    xs_resized = [_resize_to_match(x, (max_h, max_w)) for x in xs]
    return torch.cat(xs_resized, dim=dim)
# -------------------------
# CNN wrapper (builds feature extractor + output head) with optional multi-scale fusion
# -------------------------
class CNN(nn.Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 output_head_builder: Callable[[Tuple[int, int, int]], nn.Module],
                 layers: Sequence[Layer],
                 optimizer_fn: Optional[Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]] = None,
                 loss_type: str = "bce",
                 class_weight: Optional[List[float]] = None,
                 load_if_exist: bool = True,
                 logs_dir: str = "./logs/train_data",
                 checkpoint_dir: str = "./checkpoints",
                 # ====== NEW: fusion config ======
                 fusion: bool = False,
                 fusion_pos: str = "early",      # "early" | "mid" | "late" | "all"
                 fusion_type: str = "add",        # "add" | "concat" | "attention" | "mix"
                 num_streams: int = 1) -> None:
        super().__init__()
        self.input_shape = tuple(input_shape)            # (C,H,W) for a single stream
        self.output_head_builder = output_head_builder
        self.layers: List[Layer] = list(layers) if layers is not None else []
        self.load_if_exist = load_if_exist
        self.logs_dir = logs_dir
        self.checkpoint_dir = checkpoint_dir
        self.loss_type = loss_type
        self.class_weight = class_weight
        
        # Fusion config
        self.fusion = fusion and (num_streams > 1)
        self.fusion_pos = fusion_pos
        self.fusion_type = fusion_type
        self.num_streams = num_streams

                # ---- Lock fusion choice at construction time ----
        self._chosen_pos = None
        self._chosen_ftype = None
        if self.fusion:
            # lock position
            if self.fusion_pos == "all":
                self._chosen_pos = np.random.choice(["early", "mid", "late"])
            else:
                self._chosen_pos = self.fusion_pos
            # lock type
            if self.fusion_type == "mix":
                self._chosen_ftype = np.random.choice(["add", "concat", "attention"])
            else:
                self._chosen_ftype = self.fusion_type
            # cache tuple for repr/hash
            self._fusion_choice = (self._chosen_pos, self._chosen_ftype)

        self.optimizer_fn = optimizer_fn or (lambda params: torch.optim.Adam(params))

        # ========== Loss  ==========
        if loss_type == "ce":
            self.loss_fn = lambda: nn.CrossEntropyLoss()
        elif loss_type == "weighted":
            if class_weight is None:
                raise ValueError("loss_type='weighted'  class_weight")
            weight_tensor = torch.tensor(class_weight, dtype=torch.float32)
            self._class_weight = weight_tensor
            self.loss_fn = lambda: nn.CrossEntropyLoss(weight=self._class_weight)
        elif loss_type == "focal":
            self.loss_fn = lambda: FocalLoss(alpha=1.0, gamma=2.0)
        elif loss_type == "bce":
            if class_weight is None:
                pos_w = torch.tensor([1.0], dtype=torch.float32)
            else:
                pos_w = torch.tensor([class_weight[1] / class_weight[0]], dtype=torch.float32)
            self._pos_weight = pos_w
            self.loss_fn = lambda: nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)

        else:
            raise ValueError(f" loss_type: {loss_type}")
        # ==================================

        self.hash = self.generate_hash()
        self.checkpoint_filepath = os.path.join(
            self.checkpoint_dir, f"model_{self.hash}", f"model_{self.hash}.pt"
        )

        # Built modules
        self.features: nn.Module = None               # single-stream features OR post-fusion tail
        self.classifier: nn.Module = None
        # For fusion layouts
        self.branches: nn.ModuleList = nn.ModuleList()   # used for mid/late fusion
        self.mid_head: nn.Module = None                  # tail after mid fusion
        self.att_fuser: Optional[_AttentionFuse] = None

        self._built = False

    # ---- small helpers ----
    def _build_seq(
        self,
        in_ch: int,
        layer_slice: Sequence[Layer],
        hw: Tuple[int, int] = None,   # ← ： (H, W)
    ) -> Tuple[nn.Module, Tuple[int,int,int]]:
        modules: List[nn.Module] = []
        c = in_ch
        for desc in layer_slice:
            if isinstance(desc, SkipLayer):
                block = _SkipBlock(in_ch=c, f1=desc.feature_size1, f2=desc.feature_size2,
                                kernel=desc.kernel, stride=desc.stride, padding_mode=desc.convolution)
                modules.append(block)
                c = desc.feature_size2
            elif isinstance(desc, PoolingLayer):
                modules.append(_Pooling(desc.pooling_type, desc.kernel, desc.stride))
        seq = nn.Sequential(*modules) if modules else nn.Identity()

        # --- ： hw， ---
        with torch.no_grad():
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = 'cpu'
            h0 = self.input_shape[1] if hw is None else hw[0]
            w0 = self.input_shape[2] if hw is None else hw[1]
            dummy = torch.zeros(1, in_ch, h0, w0, device=device)
            out = seq(dummy)
        return seq, (out.shape[1], out.shape[2], out.shape[3])


    def _choose_fusion_type(self) -> str:
        if self.fusion_type == "mix":
            return np.random.choice(["add", "concat", "attention"])  # simple random
        return self.fusion_type

    # ---- building ----
    def _build_single_stream(self):
        # original single pipeline
        self.features, feat_shape = self._build_seq(self.input_shape[0], self.layers)
        self.classifier = self.output_head_builder(feat_shape)

    def _build_with_fusion(self):
        """（1D/2D） MLP  CNN """
        pos, ftype = self._chosen_pos, self._chosen_ftype

        # 
        if not hasattr(self, "input_shapes") or not self.input_shapes:
            #  input_shape 
            self.input_shapes = [self.input_shape] * self.num_streams

        # （True=image, False=vector）
        self.is_image_streams = []
        for shape in self.input_shapes:
            if len(shape) == 3 and shape[1] > 1 and shape[2] > 1:
                self.is_image_streams.append(True)
            else:
                self.is_image_streams.append(False)

        # ===== Early Fusion =====
        if pos == "early":
            # early 
            first_shape = self.input_shapes[0]
            if all(self.is_image_streams):
                # 
                in_ch = first_shape[0] * (self.num_streams if ftype == "concat" else 1)
                self.features, feat_shape = self._build_seq(in_ch, self.layers)
                self.classifier = self.output_head_builder(feat_shape)
            else:
                # （）
                input_dim = sum([s[0] for s, isimg in zip(self.input_shapes, self.is_image_streams) if not isimg])
                self.features = _MLPBlock(input_dim, hidden_dim=128, output_dim=64)
                self.classifier = nn.Sequential(nn.Linear(64, 1))
            if ftype == "attention":
                self.att_fuser = _AttentionFuse(self.num_streams)

        # ===== Mid Fusion =====
        elif pos == "mid":
            mid = max(1, len(self.layers)//2)
            head_desc = self.layers[:mid]
            tail_desc = self.layers[mid:]
            heads, out_shapes = [], []

            for i in range(self.num_streams):
                shape = self.input_shapes[i]
                if self.is_image_streams[i]:
                    seq, shp = self._build_seq(shape[0], head_desc)
                else:
                    seq = _MLPBlock(shape[0], hidden_dim=128, output_dim=64)
                    shp = (64, 1, 1)
                heads.append(seq)
                out_shapes.append(shp)

            self.branches = nn.ModuleList(heads)

            # === ： mid fusion  ===
            min_h = min(s[1] for s in out_shapes)
            min_w = min(s[2] for s in out_shapes)

            c_tail_in = out_shapes[0][0] * (self.num_streams if ftype == "concat" else 1)

            # tail build 
            self.features, feat_shape = self._build_seq(
                c_tail_in, tail_desc, hw=(min_h, min_w)
            )

            self.classifier = self.output_head_builder(feat_shape)

            if ftype == "attention":
                self.att_fuser = _AttentionFuse(self.num_streams)


        # ===== Late Fusion =====
        elif pos == "late":
            heads, out_shapes = [], []
            for i in range(self.num_streams):
                shape = self.input_shapes[i]
                if self.is_image_streams[i]:
                    seq, shp = self._build_seq(shape[0], self.layers)
                else:
                    seq = _MLPBlock(shape[0], hidden_dim=128, output_dim=64)
                    shp = (64, 1, 1)
                heads.append(seq)
                out_shapes.append(shp)
            self.branches = nn.ModuleList(heads)

            if ftype == "concat":
                c_sum = sum(s[0] for s in out_shapes)
                if any(self.is_image_streams):
                    self.features = nn.Identity()
                    self.classifier = self.output_head_builder((c_sum, out_shapes[0][1], out_shapes[0][2]))
                else:
                    self.features = nn.Identity()
                    self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c_sum, 1))
            else:
                if any(self.is_image_streams):
                    c, h, w = out_shapes[0]
                    self.features = nn.Identity()
                    self.classifier = self.output_head_builder((c, h, w))
                    self.classifiers = nn.ModuleList([self.output_head_builder((c, h, w)) for _ in range(self.num_streams)])
                else:
                    self.features = nn.Identity()
                    self.classifier = nn.Sequential(nn.Linear(64, 1))
                    self.classifiers = nn.ModuleList([nn.Sequential(nn.Linear(64, 1)) for _ in range(self.num_streams)])
                if ftype == "attention":
                    self.att_fuser = _AttentionFuse(self.num_streams)
        else:
            raise ValueError(f"Unknown fusion_pos: {self.fusion_pos}")

        # ----  fusion  ----
        #  "mix"， add/concat/attention
        if ftype not in ("add", "concat", "attention"):
            ftype = "attention"
        self._fusion_choice = (pos, ftype)

        #  attention  fuser，
        if ftype == "attention" and self.att_fuser is None:
            self.att_fuser = _AttentionFuse(self.num_streams)

    def _build_from_descriptors(self):
        if not self.fusion:
            self._build_single_stream()
        else:
            self._build_with_fusion()
        self._built = True

    def generate(self) -> "CNN":
        if not self._built:
            SkipLayer.GROUP_NUMBER = 1
            self._build_from_descriptors()
            SkipLayer.GROUP_NUMBER = 1
        return self

    # ---- training / eval helpers ----
    def _prepare_inputs(self, data_x: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]], device: str):
        """Accepts either (N,C,H,W)/(N,H,W) tensor/ndarray, or a list of K such arrays for multi-stream.
        Returns: list_of_tensors if fusion, else single tensor.
        """
        def to_four_d(a):
            x = torch.as_tensor(a, dtype=torch.float32)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            return x

        if self.fusion:
            if isinstance(data_x, (list, tuple)):
                xs = [to_four_d(xx).to(device) for xx in data_x]
            else:
                # if packed as (N, K, H, W), split along channel 1 if C==K and original C==1
                x = to_four_d(data_x)
                if x.shape[1] == self.num_streams:
                    xs = [x[:, i:i+1, ...].to(device) for i in range(self.num_streams)]
                else:
                    raise ValueError("Fusion enabled but data_x is single tensor without stream dimension. Provide list or (N,K,H,W)")
            return xs
        else:
            x = to_four_d(data_x).to(device)
            return x

    def evaluate(self, data, batch_size=64, device="cpu"):
        self.eval()
        self.to(device)

        x = data.get("x_test_multi", data.get("x_test"))
        y = torch.as_tensor(data["y_test"], dtype=torch.long).to(device)

        if self.fusion:
            xs = self._prepare_inputs(x, device)
            n = xs[0].size(0)
        else:
            x = self._prepare_inputs(x, device)
            n = x.size(0)

        # create dataset / loader
        if self.fusion:
            ds = torch.utils.data.TensorDataset(*(xs + [y]))
        else:
            ds = torch.utils.data.TensorDataset(x, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


        loss_fn = self.loss_fn()
        if hasattr(loss_fn, "weight") and loss_fn.weight is not None:
            loss_fn.weight = loss_fn.weight.to(device)
        if hasattr(loss_fn, "pos_weight") and getattr(loss_fn, "pos_weight", None) is not None:
            loss_fn.pos_weight = loss_fn.pos_weight.to(device)

        all_preds, all_labels = [], []
        total_loss, total = 0.0, 0

        with torch.no_grad():
            for batch in dl:
                if self.fusion:
                    *xb_list, yb = batch
                    logits = self.forward(xb_list)
                else:
                    xb, yb = batch
                    logits = self.forward(xb)
                #  yb  logits 
                if logits.ndim == 2 and logits.shape[1] == 1 and yb.ndim == 1:
                    yb = yb.unsqueeze(1).float()
                elif logits.ndim == 2 and logits.shape[1] > 1:
                    yb = yb.long()
                loss = loss_fn(logits, yb)
                total_loss += loss.item() * yb.size(0)
                if self.loss_fn().__class__.__name__ == "BCEWithLogitsLoss":
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).long().squeeze(1)
                else:
                    preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
                total += yb.size(0)

        avg_loss = total_loss / max(1, total)

        #  average
        num_classes = len(np.unique(all_labels))
        avg_type = "binary" if num_classes == 2 else "macro"

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average=avg_type, zero_division=0)
        rec  = recall_score(all_labels, all_preds, average=avg_type, zero_division=0)
        f1   = f1_score(all_labels, all_preds, average=avg_type, zero_division=0)

        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        return avg_loss, metrics


    def train_model(self, data: Dict[str, Any], batch_size: int = 64, epochs: int = 1,
                device: str = "cpu", val_split: float = 0.2):

        os.makedirs(os.path.dirname(self.checkpoint_filepath), exist_ok=True)

        # ---- ， ----
        if self.load_if_exist and os.path.exists(self.checkpoint_filepath):
            try:
                state_dict = torch.load(self.checkpoint_filepath, map_location=device)
                self.load_state_dict(state_dict)
                self.to(device)
                return {
                    "f1": 0.0, "accuracy": 0.0,
                    "precision": 0.0, "recall": 0.0
                }
            except Exception as e:
                print(f"⚠️ ，: {e}")

        self.to(device)
        self.train()

        # ----  ----
        x = data.get("x_train_multi", data.get("x_train"))
        y = torch.as_tensor(data["y_train"], dtype=torch.long)

        if self.fusion:
            xs = self._prepare_inputs(x, device)
            n = xs[0].size(0)
        else:
            x = self._prepare_inputs(x, device)
            n = x.size(0)

        y = y.to(device)

        # ----  ----
        n_val = int(n * val_split)
        use_val = (n_val > 0 and n_val < n)

        idx = torch.randperm(n, device=device)
        if use_val:
            val_idx = idx[:n_val]
            train_idx = idx[n_val:]
        else:
            val_idx = None
            train_idx = idx

        if self.fusion:
            train_ds = torch.utils.data.TensorDataset(*[xx[train_idx] for xx in xs], y[train_idx])
            val_ds   = None if not use_val else torch.utils.data.TensorDataset(*[xx[val_idx] for xx in xs], y[val_idx])
        else:
            train_ds = torch.utils.data.TensorDataset(x[train_idx], y[train_idx])
            val_ds   = None if not use_val else torch.utils.data.TensorDataset(x[val_idx], y[val_idx])

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_dl   = None if not use_val else torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # ----  ----
        opt = self.optimizer_fn(self.parameters())
        loss_fn = self.loss_fn()
        if hasattr(loss_fn, "weight") and getattr(loss_fn, "weight", None) is not None:
            loss_fn.weight = loss_fn.weight.to(device)
        if hasattr(loss_fn, "pos_weight") and getattr(loss_fn, "pos_weight", None) is not None:
            loss_fn.pos_weight = loss_fn.pos_weight.to(device)

        # ----  ----
        best_val_f1   = -1.0
        best_val_acc  = 0.0
        best_val_prec = 0.0
        best_val_rec  = 0.0

        # ===========================
        #       Training Loop
        # ===========================
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0

            with tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", leave=True) as pbar:
                for batch in pbar:
                    if self.fusion:
                        *xb_list, yb = batch
                        logits = self.forward(xb_list)
                    else:
                        xb, yb = batch
                        logits = self.forward(xb)

                    # 
                    if logits.ndim == 2 and logits.shape[1] == 1 and yb.ndim == 1:
                        yb = yb.unsqueeze(1).float()
                    elif logits.ndim == 2 and logits.shape[1] > 1:
                        yb = yb.long()

                    loss = loss_fn(logits, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    running_loss += loss.item()
                    pbar.set_postfix({"loss": f"{running_loss/(pbar.n+1):.4f}"})

            # ===========================
            #     Validation
            # ===========================
            if use_val and val_dl is not None:
                self.eval()
                all_preds, all_labels = [], []

                with torch.no_grad():
                    for batch in val_dl:
                        if self.fusion:
                            *xb_list, yb = batch
                            logits = self.forward(xb_list)
                        else:
                            xb, yb = batch
                            logits = self.forward(xb)

                        if self.loss_fn().__class__.__name__ == "BCEWithLogitsLoss":
                            probs = torch.sigmoid(logits)
                            preds = (probs > 0.5).long().squeeze(1)
                        else:
                            preds = logits.argmax(dim=1)

                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(yb.cpu().numpy())

                # ===  ===
                if len(all_labels) > 0:
                    num_classes = len(np.unique(all_labels))
                    avg_type = "binary" if num_classes == 2 else "macro"

                    val_acc  = accuracy_score(all_labels, all_preds)
                    val_prec = precision_score(all_labels, all_preds, average=avg_type, zero_division=0)
                    val_rec  = recall_score(all_labels, all_preds, average=avg_type, zero_division=0)
                    val_f1   = f1_score(all_labels, all_preds, average=avg_type, zero_division=0)

                    # ===  ===
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_val_acc = val_acc
                        best_val_prec = val_prec
                        best_val_rec = val_rec
                        torch.save(self.state_dict(), self.checkpoint_filepath)

        # ---- ， ----
        if not os.path.exists(self.checkpoint_filepath):
            torch.save(self.state_dict(), self.checkpoint_filepath)

        # ----  ----
        try:
            state_dict = torch.load(self.checkpoint_filepath, map_location=device)
            self.load_state_dict(state_dict)
            self.to(device)
        except RuntimeError as e:
            print(f"⚠️ : {e}")

        # ---- （ GA fitness） ----
        if use_val and best_val_f1 >= 0:
            return {
                "f1": best_val_f1,
                "accuracy": best_val_acc,
                "precision": best_val_prec,
                "recall": best_val_rec
            }
        else:
            # 、 val 
            return {
                "f1": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }



    # ---- nn.Module API ----
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if not self._built:
            raise RuntimeError("Call generate() before forward()")

        # （）
        if not self.fusion:
            if isinstance(x, (list, tuple)):
                x = x[0]
            if x.ndim == 2:
                z = self.features(x)
            elif x.ndim == 4:
                z = self.features(x)
            else:
                z = self.features(x.view(x.size(0), -1))
            z = self.classifier(z)
            return z

        pos, ftype = self._fusion_choice
        xs: List[torch.Tensor] = x
        n_streams = len(xs)

        # ========== Early Fusion ==========
        if pos == "early":
            # ✅ :  4D
            xs_aligned = []
            for x in xs:
                if x.ndim == 2:      # (B, L)
                    x = x.unsqueeze(1).unsqueeze(2)  # → (B, 1, 1, L)
                elif x.ndim == 3:    # (B, H, W)
                    x = x.unsqueeze(1)               # → (B, 1, H, W)
                xs_aligned.append(x)

            if ftype == "add":
                fused = _fuse_add(xs_aligned)
            elif ftype == "concat":
                fused = _fuse_concat(xs_aligned, dim=1)
            else:
                if self.att_fuser is None:
                    # ：，
                    self.att_fuser = _AttentionFuse(len(xs_aligned))
                fused = self.att_fuser(xs_aligned)

            z = self.features(fused)
            z = self.classifier(z)
            return z
        # ========== Mid Fusion ==========
        if pos == "mid":
            head_outs = []
            for i, (head, xx) in enumerate(zip(self.branches, xs)):
                if self.is_image_streams[i]:
                    out = head(xx)
                else:
                    out = head(xx.view(xx.size(0), -1))
                    out = out.unsqueeze(-1).unsqueeze(-1)
                head_outs.append(out)

            # 
            min_h = min(f.shape[2] for f in head_outs)
            min_w = min(f.shape[3] for f in head_outs)
            head_outs = [f[:, :, :min_h, :min_w] for f in head_outs]

            if ftype == "add":
                fused = _fuse_add(head_outs)
            elif ftype == "concat":
                fused = _fuse_concat(head_outs, dim=1)
            else:
                if self.att_fuser is None:
                    self.att_fuser = _AttentionFuse(len(head_outs))
                fused = self.att_fuser(head_outs)
            z = self.features(fused)
            z = self.classifier(z)
            return z

        # ========== Late Fusion ==========
        if pos == "late":
            feats = []
            for i, (head, xx) in enumerate(zip(self.branches, xs)):
                if self.is_image_streams[i]:
                    out = head(xx)
                else:
                    out = head(xx.view(xx.size(0), -1))
                    out = out.unsqueeze(-1).unsqueeze(-1)
                # ✅ flatten 
                out = out.view(out.size(0), -1)
                feats.append(out)

            # 
            if ftype == "concat":
                fused = torch.cat(feats, dim=1)
                z = self.classifier(fused)
                return z
            else:
                logits_list = [clf(f) for clf, f in zip(self.classifiers, feats)]
                if ftype == "add":
                    logits = _fuse_add(logits_list)
                else:
                    if self.att_fuser is None:
                        self.att_fuser = _AttentionFuse(len(logits_list))
                    logits = self.att_fuser(logits_list)
                if logits.ndim > 2:
                    logits = logits.view(logits.size(0), -1)
                if 'logits' in locals() and logits.ndim > 2:
                    logits = logits.view(logits.size(0), -1)
                return logits


        raise RuntimeError("Invalid fusion configuration")



    # ---- utilities ----
    def generate_hash(self) -> str:
        base = "-".join(map(str, self.layers))
        if self.fusion:
            # prefer locked choice if available
            if hasattr(self, "_fusion_choice") and self._fusion_choice is not None:
                pos, ftype = self._fusion_choice
            else:
                # fallback (shouldn't happen after __init__ lock)
                pos = self.fusion_pos if self.fusion_pos != "all" else "early"
                ftype = self.fusion_type if self.fusion_type != "mix" else "add"

            # show source only when user used all/mix
            src_info = ""
            if self.fusion_pos == "all" or self.fusion_type == "mix":
                src_info = f" (src={self.fusion_pos}:{self.fusion_type})"

            base += f"|F[{pos}:{ftype}]{src_info}"
        return base

    def __repr__(self) -> str:
        return self.generate_hash()

def get_layer_from_string(layer_definition: str) -> List[Layer]:
    """
    Convert string like '128-64-mean-max-32-32' into descriptors.
    """
    layers_str = layer_definition.split("-")
    layers: List[Layer] = []
    i = 0
    while i < len(layers_str):
        token = layers_str[i]
        if token.isdigit():
            f1 = int(token)
            f2 = int(layers_str[i + 1])
            layers.append(SkipLayer(f1, f2))
            i += 2
        else:
            layers.append(PoolingLayer(token))
            i += 1
    return layers
