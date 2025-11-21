import pickle
import numpy as np
import torch
import warnings
from gamma_cnn import AutoCNN

warnings.filterwarnings("ignore", category=FutureWarning)


def load_compact_dataset(pkl_path):
    """Load pre-split train/test dataset from pickle file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    train = data["train"]
    test  = data["test"]

    return (
        train["profile"], train["dm"], train["subband"], train["subint"], train["pdm"], train["diag"], train["label"],
        test["profile"],  test["dm"],  test["subband"],  test["subint"],  test["pdm"],  test["diag"],  test["label"]
    )


# -----------------------------
# 1. Load dataset
# -----------------------------
pkl_path = "HTRU.pkl"

(train_profile, train_dm, train_sb, train_si, train_pdm, train_diag, train_y,
 test_profile,  test_dm,  test_sb,  test_si,  test_pdm,  test_diag,  test_y) = load_compact_dataset(pkl_path)

print("✅ Data loaded successfully!")
print(f"Train set: {train_dm.shape[0]} | Test set: {test_dm.shape[0]}")


# -----------------------------
# 2. Build dataset. e.g. 1.single-modal: subband 2.multimodal fusion: profile + DM + subbands + subints)
# -----------------------------

dataset_SB = {
    "x_train_multi": train_sb,
    "y_train": train_y,
    "x_test_multi":test_sb,
    "y_test": test_y
}


dataset_multi = {
    "x_train_multi": [train_profile, train_dm, train_sb, train_si],
    "y_train": train_y,
    "x_test_multi":  [test_profile,  test_dm,  test_sb,  test_si],
    "y_test": test_y
}

print("Input shape of first modality:", dataset_multi["x_train_multi"][0].shape)


# -----------------------------
# 3. Initialize Genetic Algorithm (GA)
# -----------------------------
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

ga_sb = AutoCNN(
    population_size=10,
    maximal_generation_number=10,
    dataset=dataset_sb,
    epoch_number=10,
    batch_size=64,
    loss_type="bce",             # "ce"/"weighted"/"focal"/"bce"
    # class_weight=[1.0, train_np_p],
    fitness_cache='20251114resultsHTRU/fitness_cache_sb42.json',
    checkpoint_dir='20251114resultsHTRU/checkpoints_sb42',
    seed = SEED,
    logger_path="20251114resultsHTRU/single_sb_stats42.json",
    fusion=False,                # fusion?
    device=device
)

ga_4M = AutoCNN(
    population_size=20,
    maximal_generation_number=10,
    dataset=dataset_multi,
    epoch_number=10,
    batch_size=32,
    device=device,
    loss_type="bce",           # "ce" / "weighted" / "focal"
    fusion=True,
    fusion_pos="late",         # "early" / "mid" / "late" / "all"
    fusion_type="attention",   # "add" / "concat" / "attention" / "mix"
)

# -----------------------------
# 5. Run Evolution Process
# -----------------------------
best_sb= ga_sb.run()
print("✅ Evolution completed!")
print("Best structure:", "-".join(map(str, best_sb.layers)))
print("Fusion configuration:", getattr(best_cnn_sb, "_fusion_choice", None))


best_cnn = ga_4M.run()
print("✅ Evolution completed!")
print("Best structure:", "-".join(map(str, best_cnn.layers)))
print("Fusion configuration:", getattr(best_cnn, "_fusion_choice", None))
