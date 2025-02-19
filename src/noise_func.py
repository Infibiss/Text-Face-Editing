import math
import torch

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def noise_regularize(noises):
    loss = 0  # Инициализация потерь

    for noise in noises:  # Проход по каждому шуму в списке
        size = noise.shape[2]

        while True:  # Цикл для уменьшения размера шума
            # Вычисление лосса на основе свертки шума с его сдвинутой версией
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)  # Сдвиг по ширине
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)  # Сдвиг по высоте
            )

            if size <= 8:
                break

            # Уменьшение размера шума вдвое
            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize(noises):
    for noise in noises:
        mean = noise.mean()  # Вычисление среднего значения
        std = noise.std()  # Вычисление стандартного отклонения

        # Нормализация: вычитание среднего и деление на стандартное отклонение
        noise.data.add_(-mean).div_(std)


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength
    return latent + noise