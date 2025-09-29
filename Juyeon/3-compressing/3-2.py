import sys, os, math, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, utils
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET   = "freyface"  
MODEL     = "vae"       
LATENT    = 16      
BETA      = 1.0        
EPOCHS    = 50
BATCH     = 128
LR        = 2e-3
IMG_SIZE  = 64
OUT_DIR   = "./runs_faces"
DATA_DIR  = "./data"

# 재현성
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)


# 데이터셋
class FreyFace(Dataset):
    def __init__(self, root, transform=None):
        path_npy = os.path.join(root, "frey_faces.npy")
        if not os.path.exists(path_npy):
            # frey_faces.npy 없으면 Olivetti 대체 생성
            try:
                from sklearn.datasets import fetch_olivetti_faces
            except Exception as e:
                raise RuntimeError("scikit-learn 필요") from e
            faces = fetch_olivetti_faces(shuffle=True, download_if_missing=True) 
            arr = (faces.images * 255).astype('uint8')
            np.save(path_npy, arr)
            print(f"[INFO] {arr.shape} -> {path_npy} 저장 완료")

        arr = np.load(path_npy)  # (N, H, W)
        if arr.ndim != 3:
            raise ValueError("frey_faces.npy는 (N,H,W) 형태여야 합니다.")
        self.imgs = arr
        self.transform = transform

    def __len__(self): return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx].astype(np.uint8), mode="L")
        if self.transform: img = self.transform(img)
        return img, 0

def get_transforms(gray=False, size=64):
    if gray:
        return transforms.Compose([
            transforms.Resize(size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1,1]
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

def get_dataloader(name, img_size=64, batch_size=128):
    if name == "freyface":
        tfm = get_transforms(gray=True, size=img_size)
        ds = FreyFace(DATA_DIR, transform=tfm)
        C = 1
    elif name == "celeba":
        tfm = get_transforms(gray=False, size=img_size)
        ds = datasets.CelebA(root=DATA_DIR, split="train", target_type="attr",
                             download=True, transform=tfm)
        C = 3
    elif name == "oxfordpets":
        tfm = get_transforms(gray=False, size=img_size)
        ds = datasets.OxfordIIITPet(root=DATA_DIR, split="train",
                                    download=True, transform=tfm)
        C = 3
    else:
        raise ValueError("DATASET은 freyface | celeba | oxfordpets 중 하나여야 합니다.")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader, C

loader, C = get_dataloader(DATASET, IMG_SIZE, BATCH)
print(f"[INFO] dataset={DATASET}, channels={C}, batches={len(loader)}")


# 모델 (Conv AE / VAE)
class Encoder(nn.Module):
    def __init__(self, in_ch, latent):
        super().__init__()
        ch = 64
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1), nn.LeakyReLU(0.2, True),   # 64->32
            nn.Conv2d(ch, ch*2, 4, 2, 1), nn.BatchNorm2d(ch*2), nn.LeakyReLU(0.2, True), # 32->16
            nn.Conv2d(ch*2, ch*4, 4, 2, 1), nn.BatchNorm2d(ch*4), nn.LeakyReLU(0.2, True), # 16->8
            nn.Conv2d(ch*4, ch*8, 4, 2, 1), nn.BatchNorm2d(ch*8), nn.LeakyReLU(0.2, True), # 8->4
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(ch*8*4*4, latent)      # AE용
        self.fc_mu = nn.Linear(ch*8*4*4, latent)   # VAE용
        self.fc_logvar = nn.Linear(ch*8*4*4, latent)

    def forward(self, x, vae=False):
        h = self.net(x); h = self.flatten(h)
        if not vae: return self.fc(h)
        mu = self.fc_mu(h); logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, out_ch, latent):
        super().__init__()
        ch = 64
        self.fc = nn.Linear(latent, ch*8*4*4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ch*8, ch*4, 4, 2, 1), nn.BatchNorm2d(ch*4), nn.ReLU(True),  # 4->8
            nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1), nn.BatchNorm2d(ch*2), nn.ReLU(True),  # 8->16
            nn.ConvTranspose2d(ch*2, ch,   4, 2, 1), nn.BatchNorm2d(ch),   nn.ReLU(True),  # 16->32
            nn.ConvTranspose2d(ch, out_ch, 4, 2, 1), nn.Tanh()                                  # 32->64
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 64*8, 4, 4)
        return self.net(h)

def reparameterize(mu, logvar):
    std = (0.5*logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps*std

enc = Encoder(C, LATENT).to(DEVICE)
dec = Decoder(C, LATENT).to(DEVICE)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)


# 손실 & 유틸
def recon_loss(xhat, x):
    return F.l1_loss(xhat, x)

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def denorm(x):  
    x = (x + 1) / 2
    return x.clamp(0,1)

def show_grid(tensor, nrow=8, title=None):
    with torch.no_grad():
        grid = utils.make_grid(tensor.cpu(), nrow=nrow)
        plt.figure(figsize=(nrow, nrow)); plt.axis("off")
        if title: plt.title(title)
        plt.imshow(np.transpose(grid.numpy(), (1,2,0))); plt.show()

def interpolate(z1, z2, steps=8, slerp=True):
    if slerp:
        def _slerp(a, b, t):
            a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
            b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
            omega = torch.acos((a_n*b_n).sum(-1)).unsqueeze(-1)
            so = torch.sin(omega)
            t = t.view(-1,1)
            return (torch.sin((1-t)*omega)/so)*a + (torch.sin(t*omega)/so)*b
        ts = torch.linspace(0,1,steps, device=z1.device)
        zs = torch.stack([_slerp(z1, z2, t) for t in ts], dim=0)
    else:
        ts = torch.linspace(0,1,steps, device=z1.device).view(-1,1)
        zs = (1-ts)*z1 + ts*z2
    return zs
    
    
# 학습
for epoch in range(1, EPOCHS+1):
    enc.train(); dec.train()
    losses = []
    t0 = time.time()
    for x, _ in loader:
        x = x.to(DEVICE)
        if MODEL == "ae":
            z = enc(x, vae=False)
            xhat = dec(z)
            loss = recon_loss(xhat, x)
        else:
            mu, logvar = enc(x, vae=True)
            z = reparameterize(mu, logvar)
            xhat = dec(z)
            r = recon_loss(xhat, x)
            k = kl_loss(mu, logvar)
            loss = r + BETA * k
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    dt = time.time() - t0
    print(f"Epoch {epoch:03d}/{EPOCHS} | loss={np.mean(losses):.4f} | {dt:.1f}s")

    # 시각화 
    enc.eval(); dec.eval()
    with torch.no_grad():
        # 재구성
        x,_ = next(iter(loader)); x = x[:16].to(DEVICE)
        if MODEL == "ae":
            z = enc(x, vae=False)
        else:
            mu, logvar = enc(x, vae=True)
            z = mu
        xhat = dec(z)
        show_grid(torch.cat([denorm(x), denorm(xhat)], dim=0),
                  nrow=16, title=f"Recon (epoch {epoch})")

        if MODEL == "vae":
            z_prior = torch.randn(16, LATENT, device=DEVICE)
            x_gen = dec(z_prior)
            show_grid(denorm(x_gen), nrow=8, title=f"Prior Samples (epoch {epoch})")

            # 보간
            if x.size(0) >= 2:
                z1, z2 = z[0:1], z[1:2]
                zs = interpolate(z1, z2, steps=10, slerp=True).squeeze(1)
                x_interp = dec(zs)
                show_grid(denorm(x_interp), nrow=10, title=f"Interpolation (epoch {epoch})")

            steps = torch.linspace(-3, 3, 9, device=DEVICE)
            base = z[0:1].repeat(len(steps), 1)
            for dim in range(min(3, LATENT)):
                zz = base.clone()
                zz[:, dim] = steps
                x_trav = dec(zz)
                show_grid(denorm(x_trav), nrow=9, title=f"Traversal dim={dim} (epoch {epoch})")
