import torch
import torchvision.transforms as transforms
import lpips

def LPIPS():
    """
    LPIPS function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net='vgg')
    lpips_model = lpips_model.to(device)

    def func(img1, img2s):
      if img1.shape != img2s[0].shape:
        raise ValueError(f"ref and source must have the same shape(ref:{img1.shape}/source:{img2s[0].shape})")

      #img1 = transform(img1)
      img2_batch = img2s.to(device)#torch.stack(img2s).to(device) 
      img1_batch = img1.expand(img2_batch.shape[0], -1, -1, -1).to(device)

      with torch.no_grad():
        lpips_score = lpips_model(img1_batch, img2_batch).squeeze()

      best = int(torch.argmin(lpips_score).cpu().item())
      if lpips_score.ndim == 0:
          return best, float(lpips_score.cpu().item())
      else:
          return best, float(lpips_score[best].cpu())

    return func