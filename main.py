from backend.server import Server
from dependencies.swin_module import SwinTransformerUnetModule, SwinTransformerUnet
from torch.optim.adamw import AdamW
from functools import partial

unet = SwinTransformerUnet(num_classes=33)
opt_partial = partial(AdamW, lr=6e-05, eps=1e-08, betas=(0.9, 0.999), weight_decay=0.01)
swin_model = SwinTransformerUnetModule(num_classes=17, num_keypoints=16, swin_unet=unet, optimizer=opt_partial)
weights_path = 'dependencies/pelvis_model_weights.pth'

xray_server = Server(port = 8000, model= swin_model, weights_pth= weights_path, set_feed=1)
xray_server.start()
