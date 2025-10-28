# NOTE: This revision adds concise English docstrings/comments only. No functional changes.
import json
import os
import random
from typing import Dict, Callable, Iterable, Union, Tuple, Sequence, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

from cnn_structure import SkipLayer, PoolingLayer, CNN, Layer
from tqdm.auto import tqdm
import pandas as pd
from openpyxl import load_workbook

class AutoCNN:
    """ Genetic Algorithm (GA) driver that searches CNN structures and optional fusion configs. Trains/evaluates individuals and logs results to Excel. """
    def __init__(self, population_size: int,
                 maximal_generation_number: int,
                 dataset: Dict[str, Any],
                 output_head_builder: Optional[Callable[[Tuple[int, int, int]], nn.Module]] = None,
                 epoch_number: int = 1,
                 optimizer_fn: Optional[Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]] = None,
                 crossover_probability: float = .9,
                 mutation_probability: float = .2,
                 mutation_operation_distribution: Sequence[float] = None,
                 fitness_cache: str = "fitness.json",
                 logs_dir: str = "./logs/train_data",
                 checkpoint_dir: str = "./checkpoints",
                 device: str = "cpu",
                 loss_type: str = "ce",
                 class_weight: Optional[List[float]] = None,
                 batch_size: int = 32,
                 # ===== NEW: fusion switches =====
                 fusion: bool = False,
                 fusion_pos: str = "early",            # "early" / "mid" / "late" / "all"
                 fusion_type: str = "add",              # "add" / "concat" / "attention" / "mix"
                 num_streams: Optional[int] = None       # auto if None
                 ) -> None:

        self.logs_dir = logs_dir
        self.checkpoint_dir = checkpoint_dir
        self.fitness_cache = fitness_cache
        self.epoch_number = epoch_number
        self.optimizer_fn = optimizer_fn or (lambda params: torch.optim.Adam(params))
        self.dataset = dataset
        self.maximal_generation_number = maximal_generation_number
        self.population_size = population_size
        self.population: List[CNN] = []
        self.device = device
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.class_weight = class_weight

        # ========= Fusion config =========
        self.fusion = fusion
        self.fusion_pos = fusion_pos
        self.fusion_type = fusion_type
        # infer number of streams from dataset if not provided
        self.num_streams = num_streams or self._infer_num_streams(dataset)
        # if user forgot to enable fusion but dataset has multiple streams, keep default behavior (single stream)
        # If user enabled fusion but only single stream is detected, we turn it off to avoid crash
        if self.fusion and self.num_streams <= 1:
            print("[AutoCNN] fusion=True but only one stream detected. Turning fusion off.")
            self.fusion = False

        if self.fitness_cache is not None and os.path.exists(self.fitness_cache):
            with open(self.fitness_cache) as f:
                self.fitness = json.load(f)
        else:
            self.fitness = dict()

        if mutation_operation_distribution is None:
            self.mutation_operation_distribution = (.7, .1, .1, .1)
        else:
            self.mutation_operation_distribution = mutation_operation_distribution

        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability

        self.population_iteration = 0

        if output_head_builder is None:
            self.output_head_builder = self.get_output_head_builder()
        else:
            self.output_head_builder = output_head_builder

        self.input_shape = self.get_input_shape()

    # ----- shapes / heads -----
    def _infer_num_streams(self, dataset: Dict[str, Any]) -> int:
        # If user provides x_train_multi=[x1,x2,...]
        if "x_train_multi" in dataset:
            xs = dataset["x_train_multi"]
            return len(xs)
        x = dataset.get("x_train")
        if x is None:
            return 1
        # If packed as (N, K, H, W) and original channel is 1, treat K as num_streams
        if isinstance(x, np.ndarray):
            shape = x.shape
        elif torch.is_tensor(x):
            shape = tuple(x.shape)
        else:
            return 1
        if len(shape) == 4:
            c = shape[1]
            return c if c > 1 else 1
        return 1

    def get_input_shape(self) -> Tuple[int, int, int]:
        """resize 
        -    64
        -   6464
         pandas + numpy OpenCV
        """
        # =====  shape  =====
        if self.fusion and "x_train_multi" in self.dataset:
            sample = self.dataset["x_train_multi"][0]
        else:
            sample = self.dataset.get("x_train")
            if sample is None:
                raise ValueError("Dataset missing x_train or x_train_multi")

        #  numpy
        if torch.is_tensor(sample):
            sample = sample.cpu().numpy()
        shape = sample.shape[1:]

        if len(shape) == 2:
            c, h, w = 1, shape[0], shape[1]
        elif len(shape) == 3:
            c, h, w = shape
        elif len(shape) == 1:
            c, h, w = shape[0], 1, 1
        else:
            raise ValueError(f"Unsupported input shape {shape}")

        # =====  =====
        if self.fusion and "x_train_multi" in self.dataset:
            print(f"\n Detected multi-modal input ({len(self.dataset['x_train_multi'])} modalities).")
            shapes = []
            for i, arr in enumerate(self.dataset["x_train_multi"]):
                s = arr.shape[1:]
                shapes.append(s)
                print(f"   Modality {i+1}: {s}")

            # ===   1D/2D  ===
            normalized_shapes = []
            for s in shapes:
                if len(s) == 1:
                    normalized_shapes.append((64, 64))  # 
                else:
                    normalized_shapes.append((s[-2], s[-1]))

            if len(set(normalized_shapes)) > 1:
                print(" Modalities have inconsistent shapes, resizing to standard 64 / 6464 ...")
                #  resize 
                self.dataset["x_train_multi"] = [smart_resize(a) for a in self.dataset["x_train_multi"]]
                self.dataset["x_test_multi"]  = [smart_resize(a) for a in self.dataset["x_test_multi"]]
                print(" All modalities resized to 64 or 6464 using pandas interpolation.")
            else:
                print(" Modalities detected as consistent (1D treated as 6464 equivalent).")
                
                def resize_1d(arr: np.ndarray, target_len=64):
                    """ (N,L)  target_len"""
                    N, L = arr.shape
                    new_x = np.linspace(0, L - 1, target_len)
                    old_x = np.arange(L)
                    out = np.stack([
                        np.interp(new_x, old_x, row)
                        for row in arr
                    ])
                    return out.astype(np.float32)[:, None, :, None]  # (N,1,64,1)

                def resize_2d(arr: np.ndarray, target_hw=(64, 64)):
                    """ (N,H,W)  target_hw"""
                    N, H, W = arr.shape
                    th, tw = target_hw
                    out = np.zeros((N, th, tw), dtype=np.float32)
                    for i in range(N):
                        df = pd.DataFrame(arr[i])
                        # 
                        df_resampled = df.reindex(
                            np.linspace(0, H - 1, th)
                        ).interpolate(method="linear", axis=0).bfill().ffill()
                        # 
                        df_resampled = df_resampled.T.reindex(
                            np.linspace(0, W - 1, tw)
                        ).interpolate(method="linear", axis=0).bfill().ffill().T
                        out[i] = df_resampled.to_numpy(dtype=np.float32)
                    return out[:, None, :, :]  # (N,1,64,64)

                def smart_resize(arr):
                    if arr.ndim == 2:
                        return resize_1d(arr)
                    elif arr.ndim == 3:
                        return resize_2d(arr)
                    elif arr.ndim == 4 and arr.shape[1] == 1:
                        return resize_2d(arr[:, 0])
                    else:
                        return arr.astype(np.float32)

                self.dataset["x_train_multi"] = [smart_resize(a) for a in self.dataset["x_train_multi"]]
                self.dataset["x_test_multi"]  = [smart_resize(a) for a in self.dataset["x_test_multi"]]
                print(" All modalities resized to 64 or 6464 using pandas interpolation.")

        return (c, h, w)

    def get_output_head_builder(self) -> Callable[[Tuple[int, int, int]], nn.Module]:
        """
         loss_type 
        - CrossEntropyLoss:  (C=2)
        - BCEWithLogitsLoss:  (C=1)
        """
        y = self.dataset['y_train']
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        output_size = int(np.unique(y).shape[0])

        def builder(feature_shape: Tuple[int, int, int]) -> nn.Module:
            out_dim = 1 if self.loss_type == "bce" else output_size
            return nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(out_dim)
            )

        return builder
    
    def _ensure_unique_population(self, population: List["CNN"]) -> List["CNN"]:
        """
        -  hash 
        -  population_size <= 20 
        """
        if self.population_size > 20:
            return population

        unique_pop = []
        seen_hashes = set()

        for cnn in population:
            if cnn.hash not in seen_hashes:
                seen_hashes.add(cnn.hash)
                unique_pop.append(cnn)

        # 
        while len(unique_pop) < self.population_size:
            depth = random.randint(2, 5)
            layers = []
            for _ in range(depth):
                r = random.random()
                if r < .5:
                    layers.append(self.random_skip())
                else:
                    layers.append(self.random_pooling())
            new_cnn = self.generate_cnn(layers)
            if new_cnn.hash not in seen_hashes:
                seen_hashes.add(new_cnn.hash)
                unique_pop.append(new_cnn)
        if len(unique_pop) < len(population):
            print(f"[GA] Duplicates removed: {len(population) - len(unique_pop)} | Final unique population size = {len(unique_pop)}")
        return unique_pop
    
    # ----- population init -----
    def initialize(self) -> None:
        self.population.clear()
        for _ in range(self.population_size):
            depth = random.randint(2, 5)
            layers: List[Layer] = []
            for _ in range(depth):
                r = random.random()
                if r < .5:
                    layers.append(self.random_skip())
                else:
                    layers.append(self.random_pooling())
            cnn = self.generate_cnn(layers)
            self.population.append(cnn)
        self.population = self._ensure_unique_population(self.population)

    def random_skip(self) -> SkipLayer:
        f1 = 2 ** random.randint(4, 9)
        f2 = 2 ** random.randint(4, 9)
        return SkipLayer(f1, f2)

    def random_pooling(self) -> PoolingLayer:
        return PoolingLayer('max')
        # return PoolingLayer('max' if random.random() < .5 else 'mean')

    # ----- GA ops -----
    def evaluate_fitness(self, population: Iterable[CNN]) -> None:
        for cnn in population:
            if cnn.hash not in self.fitness:
                self.evaluate_individual_fitness(cnn)

    def evaluate_individual_fitness(self, cnn: CNN) -> None:
        import gc
        import torch

        try:
            cnn.generate()
            if self.dataset.get("x_train_multi") is not None:
                train_data = {
                    "x_train_multi": self.dataset["x_train_multi"],
                    "y_train": self.dataset["y_train"]
                }
                test_data = {
                    "x_test_multi": self.dataset["x_test_multi"],
                    "y_test": self.dataset["y_test"]
                }
            else:
                train_data = {
                    "x_train": self.dataset["x_train"],
                    "y_train": self.dataset["y_train"]
                }
                test_data = {
                    "x_test": self.dataset["x_test"],
                    "y_test": self.dataset["y_test"]
                }

            cnn.train_model(train_data, epochs=self.epoch_number, batch_size=self.batch_size, device=self.device)
            loss, metrics = cnn.evaluate(test_data, batch_size=self.batch_size, device=self.device)
            self.fitness[cnn.hash] = metrics
            tqdm.write(f"{cnn}  Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
                       f"Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

        ### debugging purposes only ###:
        # except Exception as e:
        #     tqdm.write(f"Error during individual evaluation: {e}")
        #     metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        #     self.fitness[cnn.hash] = metrics

        finally:
            if self.fitness_cache is not None:
                with open(self.fitness_cache, "w") as f:
                    json.dump(self.fitness, f)

            try:
                del cnn.features
                del cnn.classifier
                if hasattr(cnn, "branches"):
                    del cnn.branches
                if hasattr(cnn, "classifiers"):
                    del cnn.classifiers
            except Exception:
                pass

            del cnn
            gc.collect()

            # ----  MPS / CUDA  ----
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()  #  
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def select_two_individuals(self, population: Sequence[CNN]) -> CNN:
        cnn1, cnn2 = random.sample(population, 2)
        fit1 = self.fitness[cnn1.hash]["f1"]
        fit2 = self.fitness[cnn2.hash]["f1"]
        return cnn1 if fit1 > fit2 else cnn2

    def split_individual(self, cnn: CNN) -> Tuple[Sequence[Layer], Sequence[Layer]]:
        split_idx = random.randint(0, len(cnn.layers))
        return cnn.layers[:split_idx], cnn.layers[split_idx:]

    def generate_offsprings(self) -> List[CNN]:
        """ Crossover + Mutation
        """
        offsprings: List[Sequence[Layer]] = []
        while len(offsprings) < len(self.population):
            p1 = self.select_two_individuals(self.population)
            p2 = self.select_two_individuals(self.population)
            while p1.hash == p2.hash:
                p2 = self.select_two_individuals(self.population)

            # === crossover ===
            if random.random() < self.crossover_probability:
                p1_1, p1_2 = self.split_individual(p1)
                p2_1, p2_2 = self.split_individual(p2)
                offsprings.append([*p1_1, *p2_2])
                offsprings.append([*p2_1, *p1_2])
            else:
                offsprings.append(p1.layers)
                offsprings.append(p2.layers)

        # === mutation ===
        choices = ['add_skip', 'add_pooling', 'remove', 'change']
        for layers in offsprings:
            if random.random() < self.mutation_probability:
                if len(layers) == 0:
                    i = 0
                    operation = random.choice(choices[:2])
                else:
                    i = random.randint(0, len(layers) - 1)
                    operation = random.choices(choices, weights=self.mutation_operation_distribution)[0]

                if operation == 'add_skip':
                    layers.insert(i, self.random_skip())
                elif operation == 'add_pooling':
                    layers.insert(i, self.random_pooling())
                elif operation == 'remove' and len(layers) > 1:
                    layers.pop(i)
                else:
                    if isinstance(layers[i], SkipLayer):
                        layers[i] = self.random_skip()
                    else:
                        layers[i] = self.random_pooling()

        # === build CNNs ===
        offspring_cnns = []
        for layers in offsprings:
            cnn = self.generate_cnn(layers)

            #  
            if self.fusion:
                parent = random.choice(self.population)
                if hasattr(parent, "_fusion_choice"):
                    pos, ftype = parent._fusion_choice
                    cnn._chosen_pos = pos
                    cnn._chosen_ftype = ftype
                    cnn._fusion_choice = (pos, ftype)
                    #  num_streams
                    cnn.num_streams = getattr(parent, "num_streams", self.num_streams)
                    cnn.fusion = True
                    print(f" Inherited fusion from parent: pos={pos}, type={ftype}")

            offspring_cnns.append(cnn)

        return offspring_cnns

    def _is_structure_valid(self, layers: Sequence["Layer"]) -> bool:
        h, w = self.input_shape[1], self.input_shape[2]
        pool_count = 0
        for desc in layers:
            if isinstance(desc, PoolingLayer):
                pool_count += 1
                stride_h, stride_w = desc.stride
                h = h // stride_h
                w = w // stride_w
                if h < 2 or w < 2:   #  
                    return False
            elif isinstance(desc, SkipLayer):
                #  conv block 1x1 
                if h < 2 or w < 2:
                    return False
        if pool_count > int(np.log2(min(self.input_shape[1], self.input_shape[2]))):
            return False
        return True

    def generate_cnn(self, layers: Sequence["Layer"]) -> "CNN":
        """
         CNN  MLP 
         (C,1,1) MLP 
         (C,H,W) CNN 
        """
        # 
        is_vector_input = (
            self.input_shape[1] == 1 and self.input_shape[2] == 1
        )

        # ==========  MLP ==========
        if is_vector_input:
            # 
            valid_layers = [l for l in layers if isinstance(l, int)]
            if not valid_layers:
                #  layers 
                depth = random.randint(2, 4)
                valid_layers = [random.choice([32, 64, 128, 256]) for _ in range(depth)]
            #  CNN MLPBlock
            return CNN(
                input_shape=self.input_shape,
                output_head_builder=self.output_head_builder,
                layers=valid_layers,
                optimizer_fn=self.optimizer_fn,
                loss_type=self.loss_type,
                class_weight=self.class_weight,
                logs_dir=self.logs_dir,
                checkpoint_dir=self.checkpoint_dir,
                fusion=self.fusion,
                fusion_pos=self.fusion_pos,
                fusion_type=self.fusion_type,
                num_streams=self.num_streams
            )

        # ==========  CNN ==========
        #  00
        max_try = 5
        for _ in range(max_try):
            if self._is_structure_valid(layers):
                break
            # 
            depth = random.randint(2, 5)
            layers = []
            for _ in range(depth):
                r = random.random()
                if r < 0.5:
                    layers.append(self.random_skip())
                else:
                    layers.append(self.random_pooling())
        else:
            # 
            print(" Warning: too many invalid offspring, removing one random max pooling layer instead of full reset.")
            pool_indices = [i for i, l in enumerate(layers) if isinstance(l, PoolingLayer)]
            if pool_indices:
                drop_i = random.choice(pool_indices)
                print(f" Removed pooling layer at index {drop_i}")
                layers.pop(drop_i)
            else:
                print("No pooling layer found; using fallback structure 32-128.")
                layers = [SkipLayer(32, 128), PoolingLayer("max")]

        #  CNN
        return CNN(
            input_shape=self.input_shape,
            output_head_builder=self.output_head_builder,
            layers=layers,
            optimizer_fn=self.optimizer_fn,
            loss_type=self.loss_type,
            class_weight=self.class_weight,
            logs_dir=self.logs_dir,
            checkpoint_dir=self.checkpoint_dir,
            fusion=self.fusion,
            fusion_pos=self.fusion_pos,
            fusion_type=self.fusion_type,
            num_streams=self.num_streams
        )

        
    def environmental_selection(self, offsprings):
        """ Combine population and children; keep best via tournament, ensure elitism (keep best CNN). """
        whole_population = list(self.population)
        whole_population.extend(offsprings)
        
        new_population = []
        while len(new_population) < len(self.population):
            p = self.select_two_individuals(whole_population)
            new_population.append(p)

        best_cnn = max(whole_population, key=lambda x: self.fitness[x.hash]["f1"])
        print("Best CNN:", best_cnn, "Score:", self.fitness[best_cnn.hash])

        if best_cnn not in new_population:
            worst_cnn = min(new_population, key=lambda x: self.fitness[x.hash]["f1"])
            print("Worst CNN:", worst_cnn, "Score:", self.fitness[worst_cnn.hash])
            new_population.remove(worst_cnn)
            new_population.append(best_cnn)
            
        return new_population

def run(self, xlsx_path="population_log.xlsx") -> "CNN":
    """
    Main GA-CNN evolutionary loop.
    """

    print("Initializing Population")
    self.initialize()
    print("Population Initialization Done:", [repr(p) for p in self.population])

    # ---------- Generation 0 ----------
    self.evaluate_fitness(self.population)
    print("Evaluated initial population fitness.")

    # ---------- Evolution loop ----------
    for gen in range(self.maximal_generation_number):
        print(f"\n=== Generation {gen+1} ===")
        print("Evaluating Population fitness")
        self.evaluate_fitness(self.population)
        print("Evaluating Population fitness Done:", self.fitness)

        # ---- Generate offsprings ----
        print("Generating Offsprings")
        offsprings = self.generate_offsprings()
        print("Generating Offsprings Done:", offsprings)

        print("Evaluating Offsprings")
        self.evaluate_fitness(offsprings)
        print("Evaluating Offsprings Done:", self.fitness)

        # ---- Environmental selection ----
        print("Selecting new environment")
        self.population = list(self.environmental_selection(offsprings))
        print("Selecting new environment Done:", self.population)

        # ---- Find best CNN ----
        best_cnn = max(self.population, key=lambda x: self.fitness[x.hash]["f1"])
        best_f1 = self.fitness[best_cnn.hash]["f1"]
        print("Best CNN:", best_cnn, "Score:", self.fitness[best_cnn.hash])

    print("✅ All generations finished.")
    return best_cnn

