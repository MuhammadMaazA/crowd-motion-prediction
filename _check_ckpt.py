import torch, os
ckpt_dir = 'checkpoints/sdd'
for f in sorted(os.listdir(ckpt_dir)):
    if 'deathCircle' in f and f.endswith('.pt'):
        c = torch.load(os.path.join(ckpt_dir, f), map_location='cpu', weights_only=False)
        print(f"{f}: epoch={c.get('epoch','?')}  ade={c.get('ade','?'):.4f}")
