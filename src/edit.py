import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import sys
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
import argparse
import ninja

sys.path.append(os.path.join(os.path.dirname(__file__), '../data/CLIP-main'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../data/stylegan2-pytorch-master'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../data/encoder4editing-main'))

device = torch.device('cuda')
transform_size = (1024, 1024)

# <------------- ЗАГРУЗКА СТАЙЛГАНА ------------->
def stylegan_build(image_path):
    from model import Generator

    # Параметры генератора
    size = 1024  # Размер изображения
    latent_dim = 512  # Размер латентного пространства
    n_mlp = 8  # Количество слоев MLP
    channel_multiplier = 2
    ckpt = '../data/downloads/stylegan2-ffhq-config-f.pt'  # Путь к весам StyleGAN2

    # Инициализация генератора
    generator = Generator(size, latent_dim, n_mlp, channel_multiplier=channel_multiplier).to(device)
    checkpoint = torch.load(ckpt)
    generator.load_state_dict(checkpoint["g_ema"])  # Подгрузка весов
    generator.eval()  # Режим валидации

    return generator

# <------------- ВЫРАВНИВАНИЕ ЛИЦА ------------->
def align_face(image_path):
    import dlib
    from utils.alignment import align_face
    import packaging.version
    if packaging.version.parse(Image.__version__) >= packaging.version.parse('10.0.0'):
        Image.ANTIALIAS = Image.LANCZOS

    predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    if aligned_image.size != transform_size:
        aligned_image = cv2.resize(np.array(aligned_image), transform_size, interpolation=cv2.INTER_CUBIC)

    return aligned_image

# <------------- ЗАГРУЗКА ЭНКОДЕРА ------------->
def encoder_build():
    from models.psp import pSp  # Импортируем модель pSp
    from argparse import Namespace

    model_path = '../data/downloads/e4e_ffhq_encode.pt'  # Путь к весам энкодера
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path

    opts = Namespace(**opts)

    net = pSp(opts).to(device)
    net.load_state_dict(ckpt['state_dict'], strict=True)
    net.eval()

    return net

# <------------- ИНВЕРСИЯ ЭНКОДЕРА ------------->
def encoder_invert(aligned_image, net):
    # Если модель обучена на 256×256, то нужно сделать resize
    aligned_img_pil = Image.fromarray(aligned_image)

    input_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    in_tensor = input_transform(aligned_img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        image, w_plus_code = net(in_tensor, randomize_noise=False, return_latents=True)

    img_np = image[0].cpu().detach().numpy()  # shape: (3, H, W)
    img_np = np.transpose(img_np, (1, 2, 0))  # -> (H, W, 3)

    img_resized = cv2.resize(img_np, (transform_size[0], transform_size[1]), interpolation=cv2.INTER_CUBIC)

    img_resized_t = np.transpose(img_resized, (2, 0, 1))  # (3, newH, newW)
    img_resized_t = torch.from_numpy(img_resized_t).unsqueeze(0).float().to(device)  # [1, 3, newH, newW]
    image = img_resized_t

    return w_plus_code

# <------------- ШУМОВАЯ ИНВЕРСИЯ (ДОПОЛНИТЕЛЬНО) ------------->
def noise_invertion(generator, aligned_image, w_plus_code):
    import lpips  # Библиотека для вычисления perceptual loss
    from noise_func import noise_regularize, get_lr, latent_noise, noise_normalize

    transform_to_neg1_1 = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # теперь -> [-1,1]
    ])
    aligned_image_tensor = transform_to_neg1_1(aligned_image).unsqueeze(0).to(device)

    latent_w = w_plus_code.clone().to(device)
    latent_mean = latent_w.mean(dim=0, keepdim=True)
    latent_std = ((latent_w - latent_mean) ** 2).mean().sqrt()
    latent_in = latent_w.detach().clone()  # .repeat(1, generator.n_latent, 1)
    latent_in.requires_grad = True

    noises_single = generator.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(1, 1, 1, 1).normal_())
        noises[-1].requires_grad = True

    # Инверсия
    num_steps_w = 1000  # Количество шагов оптимизации в W-пространстве
    alpha = 5  # Коэффициент для перцептивной потери
    beta = 1e5  # Коэффициент для шумовой потери

    init_lr = 0.01
    optimizer_w = optim.Adam([latent_in] + noises, lr=init_lr)  # Оптимизатор для инверсии в W-пространстве
    # scheduler_w = torch.optim.lr_scheduler.ExponentialLR(optimizer_w, gamma=0.99)  # Плавное снижение скорости обучения
    mse = torch.nn.MSELoss()
    perceptual_loss = lpips.PerceptualLoss(model="net-lin", net="alex", use_gpu=device)

    # Инверсия в W-пространстве с шумовой регуляризацией
    for step in range(num_steps_w):
        torch.cuda.empty_cache()
        t = step / num_steps_w
        lr = get_lr(t, init_lr)
        optimizer_w.param_groups[0]["lr"] = lr
        optimizer_w.zero_grad()  # Обнуление градиентов перед новой итерацией

        noise_strength = latent_std * 0.05 * max(0, 1 - t / 0.75) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        generated_img, _ = generator([latent_n], input_is_latent=True, noise=noises)  # Генерация изображения из W-вектора

        # Основные потери
        noise_loss = noise_regularize(noises)
        mse_loss = mse(generated_img, aligned_image_tensor)  # Вычисление потерь MSE
        lpips_loss = perceptual_loss(generated_img, aligned_image_tensor).mean()  # Вычисление перцептивной потери
        # noise_reg_loss = noise_regularize(noises) * noise_strength  # (опционально) Шумовая регуляризация

        # Суммарные потери
        loss = mse_loss + alpha * lpips_loss + beta * noise_loss
        loss.backward()  # Обратное распространение ошибки

        optimizer_w.step()  # Обновление параметров W-вектора

        noise_normalize(noises)

        if step + 1 == num_steps_w:
            save_image(generated_img, f"z_inversion_final.png")  # Сохранение сгенерированного изображения
            return generated_img, aligned_image_tensor, latent_in

# <------------- ОПТИМИЗАЦИЯ ------------->
def optimization(generator, prompt, latent_in, aligned_image_tensor):
    import ftfy
    import clip
    from losses import IDLoss, CLIPLoss
    id_loss = IDLoss(device=device)
    clip_loss = CLIPLoss()

    text_inputs = torch.cat([clip.tokenize(prompt)]).to(device)

    latent = latent_in.detach().clone()
    latent.requires_grad = True
    target_image = aligned_image_tensor
    # target_image = transforms.ToTensor()(aligned_image).unsqueeze(0).to(device)

    optimizer = torch.optim.Adam([latent], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    num_steps = 200  # количество шагов оптимизации
    l2_lambda = 0.01  # коэффициент для L2 регуляризации
    id_lambda = 0.05  # коэффициент для identity loss

    for step in range(num_steps):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        image_gen, _ = generator([latent], input_is_latent=True, randomize_noise=False)

        clip_loss_val = clip_loss(image_gen, text_inputs)
        l2_loss_val = torch.sum((latent - latent_in) ** 2)
        id_loss_val = id_loss(image_gen, target_image)
        loss = clip_loss_val + l2_lambda * l2_loss_val + id_lambda * id_loss_val

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Визуализация и вывод потерь
        if step + 1 == num_steps:
            save_image(image_gen, f"edited_image.png")  # Сохранение сгенерированного изображения


# <------------- ПАРСИНГ АРГУМЕНТОВ ------------->
def parse_args():
    parser = argparse.ArgumentParser(description="Редактирование картинки")
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        required=True,
        help="Путь до файла с изображением (например, /home/user/pic.png)"
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default="",
        help="Текст, который нужно использовать при редактировании"
    )

    args = parser.parse_args()

    return args

# <------------- СКАЧИВАНИЕ ВЕСОВ ------------->
def download_weights():
    from download_files import download_files
    download_files("../data/downloads")

# <------------- ОСНОВНОЙ КОД ------------->
def main():
    print("Args parsing")
    args = parse_args()

    print("Downloading weights")
    download_weights()

    print("Generator building")
    generator = stylegan_build(args.image_path)

    print("Aligning image")
    aligned_image = align_face(args.image_path)

    print("Encoder building")
    net = encoder_build()

    print("Encoder inverting")
    w_plus_code = encoder_invert(aligned_image, net)

    print("Extra noise inverting")
    generated_img, aligned_image_tensor, latent_in = noise_invertion(generator, aligned_image, w_plus_code)

    print("Final optimization")
    optimization(generator, args.text, latent_in, aligned_image_tensor)

    print("Finished!")

if __name__ == "__main__":
    main()