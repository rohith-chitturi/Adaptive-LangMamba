import torch
from options.train_options import TrainOptions
from models import create_model
import traceback

def main():
    print("==================================================")
    print("PHASE A — BASELINE FORWARD-PASS VALIDATION")
    print("==================================================")
    
    # 1. Load existing configuration/model
    try:
        opt = TrainOptions().parse()
    except Exception as e:
        print(f"[FAIL] Configuration initialization failed: {e}")
        return

    # 2. Use CPU
    opt.gpu_ids = [-1]
    opt.isTrain = True
    opt.model = 'langmamba'
    opt.checkpoints_dir = './checkpoints'

    # 3. Initialize model
    print("\nAttempting to initialize LANGMAMBAModel...")
    try:
        model = create_model(opt)
        model.setup(opt)
        print("[PASS] Model initialization.")
    except Exception as e:
        print(f"[FAIL] Model initialization failed. Reason:\n{traceback.format_exc()}")
        print("Note: If blocked by missing pretrained checkpoints (e.g. VQGAN), this is expected.")
        # Try to initialize just the generator (SEED) as a synthetic test
        from models.seed import SEED
        import torchvision.models as models
        print("\nAttempting synthetic initialization of SEED (Generator)...")
        try:
            # We'll use a dummy encoder since we can't load VQGAN
            class DummyEncoder(torch.nn.Module):
                def forward(self, x):
                    return None, [x]*5 # Just return dummy features
            dummy_encoder = DummyEncoder()
            seed_model = SEED(opt, dummy_encoder)
            print("[PASS] Synthetic SEED initialization.")
            
            # Synthetic forward test
            x = torch.randn(2, 1, 256, 256)
            print(f"Input shape: {x.shape}")
            y = seed_model(x)
            print(f"Output shape: {y.shape}")
            print("[PASS] Synthetic forward validation.")
        except Exception as e2:
            print(f"[FAIL] Synthetic validation failed: {e2}")
        return

    # 4. Use small synthetic input
    print("\nRunning forward pass...")
    data = {
        'A': torch.randn(2, 1, 256, 256),
        'B': torch.randn(2, 1, 256, 256),
        'A_paths': ['dummy1', 'dummy2'],
        'B_paths': ['dummy1', 'dummy2']
    }
    
    try:
        model.set_input(data)
        model.forward()
        print(f"Input A shape: {model.data_A.shape}")
        print(f"Output shape: {model.output.shape}")
        print("[PASS] Forward validation.")
    except Exception as e:
        print(f"[FAIL] Forward pass failed. Reason:\n{traceback.format_exc()}")

if __name__ == '__main__':
    main()
