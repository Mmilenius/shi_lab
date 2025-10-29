"""
lab_shi_4_v2.py
DCGAN для генерації зображень пошкоджень доріг (варіант №7)
Автор: Maksym Lukomskyi
Дата: 2025-10-29

Функціонал:
- Автоматичне завантаження датасету "alvarobasily/road-damage" через KaggleHub
- Побудова архітектури генератора та дискримінатора
- Навчання DCGAN (Generator + Discriminator)
- Збереження результатів: графіки втрат, сітки зображень, чекпоінти, фінальна модель
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kagglehub
import zipfile

# -----------------------
# ПАРАМЕТРИ
# -----------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=False,
                   help="Шлях до директорії зображень. Якщо не вказано — буде завантажено датасет KaggleHub.")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--noise_dim", type=int, default=100)
    p.add_argument("--save_dir", type=str, default="dcgan_output")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    return p.parse_args()

# -----------------------
# МОДЕЛІ
# -----------------------
def make_generator_model(noise_dim, img_size):
    model = keras.Sequential(name="Generator")
    model.add(layers.Dense((img_size // 8) * (img_size // 8) * 256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((img_size // 8, img_size // 8, 256)))
    model.add(layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    return model

def make_discriminator_model(img_size):
    model = keras.Sequential(name="Discriminator")
    model.add(layers.Input(shape=(img_size, img_size, 3)))
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(256, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# -----------------------
# ДОПОМІЖНІ ФУНКЦІЇ
# -----------------------
def generate_and_save_images(model, epoch, test_input, save_dir, img_size):
    predictions = model(test_input, training=False)
    preds = (predictions.numpy() + 1.0) * 127.5
    preds = np.clip(preds, 0, 255).astype(np.uint8)
    n = preds.shape[0]
    grid_w = int(np.ceil(np.sqrt(n)))
    grid_h = int(np.ceil(n / grid_w))
    grid_img = Image.new('RGB', (grid_w * img_size, grid_h * img_size))
    for i in range(n):
        r, c = i // grid_w, i % grid_w
        grid_img.paste(Image.fromarray(preds[i]), (c * img_size, r * img_size))
    os.makedirs(save_dir, exist_ok=True)
    grid_img.save(os.path.join(save_dir, f"epoch_{epoch:04d}.png"))

def plot_losses(history, save_dir):
    epochs = len(history['g_loss'])
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['g_loss'], label='Generator')
    plt.plot(history['d_loss'], label='Discriminator')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(history['d_real_loss'], label='D real')
    plt.plot(history['d_fake_loss'], label='D fake')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "losses.png"))
    plt.close()

# -----------------------
# НАВЧАННЯ
# -----------------------
def train(dataset, generator, discriminator, args):
    gen_opt = keras.optimizers.Adam(args.lr, beta_1=args.beta1)
    disc_opt = keras.optimizers.Adam(args.lr, beta_1=args.beta1)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(generator_optimizer=gen_opt, discriminator_optimizer=disc_opt,
                               generator=generator, discriminator=discriminator)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)

    seed = tf.random.normal([16, args.noise_dim])
    history = {'g_loss': [], 'd_loss': [], 'd_real_loss': [], 'd_fake_loss': []}

    @tf.function
    def train_step(images):
        noise = tf.random.normal([args.batch_size, args.noise_dim])
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_imgs = generator(noise, training=True)
            real_out = discriminator(images, training=True)
            fake_out = discriminator(fake_imgs, training=True)
            real_labels = tf.ones_like(real_out) * 0.9
            fake_labels = tf.zeros_like(fake_out)
            d_real_loss = loss_fn(real_labels, real_out)
            d_fake_loss = loss_fn(fake_labels, fake_out)
            d_loss = d_real_loss + d_fake_loss
            g_loss = loss_fn(tf.ones_like(fake_out), fake_out)
        grads_g = g_tape.gradient(g_loss, generator.trainable_variables)
        grads_d = d_tape.gradient(d_loss, discriminator.trainable_variables)
        gen_opt.apply_gradients(zip(grads_g, generator.trainable_variables))
        disc_opt.apply_gradients(zip(grads_d, discriminator.trainable_variables))
        return g_loss, d_loss, d_real_loss, d_fake_loss

    print("=== Початок навчання DCGAN ===")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        g_loss_avg, d_loss_avg, d_real_avg, d_fake_avg, n = 0, 0, 0, 0, 0
        for batch in dataset:
            if batch.shape[0] != args.batch_size:
                continue
            g_loss, d_loss, d_real, d_fake = train_step(batch)
            g_loss_avg += g_loss; d_loss_avg += d_loss
            d_real_avg += d_real; d_fake_avg += d_fake; n += 1
        if n == 0:
            print("⚠️ Недостатньо зображень для batch_size.")
            return history
        history['g_loss'].append(float(g_loss_avg/n))
        history['d_loss'].append(float(d_loss_avg/n))
        history['d_real_loss'].append(float(d_real_avg/n))
        history['d_fake_loss'].append(float(d_fake_avg/n))
        if epoch % args.save_every == 0 or epoch == 1:
            generate_and_save_images(generator, epoch, seed, args.save_dir, args.img_size)
        if epoch % 5 == 0:
            manager.save()
        print(f"Епоха {epoch}/{args.epochs} | G={history['g_loss'][-1]:.4f} | D={history['d_loss'][-1]:.4f} | "
              f"час {time.time()-start:.1f}s")

    plot_losses(history, args.save_dir)
    generate_and_save_images(generator, args.epochs, seed, args.save_dir, args.img_size)
    manager.save()
    generator.save(os.path.join(args.save_dir, "generator_final.h5"))
    print("=== Навчання завершено ===")
    return history

# -----------------------
# MAIN
# -----------------------
def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Результати будуть збережені у: {args.save_dir}")

    # --- Автоматичне завантаження датасету ---
    if not args.data_dir:
        print("Директорію не вказано — завантаження датасету 'alvarobasily/road-damage'...")
        dataset_path = kagglehub.dataset_download("alvarobasily/road-damage")
        print("Path to dataset files:", dataset_path)

        # Розпаковка ZIP архівів
        for file in os.listdir(dataset_path):
            if file.endswith(".zip"):
                zip_path = os.path.join(dataset_path, file)
                print(f"Розпаковка архіву: {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)

        # Пошук директорії зображень
        for root, _, files in os.walk(dataset_path):
            if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
                args.data_dir = root
                break

        if not args.data_dir:
            raise ValueError("❌ Не знайдено зображень у датасеті!")
        else:
            print("Папка із зображеннями:", args.data_dir)

    # --- Завантаження даних ---
    print("Завантаження зображень...")
    ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        labels=None,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=True
    )

    def normalize(x): return (tf.cast(x, tf.float32) - 127.5) / 127.5
    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

    generator = make_generator_model(args.noise_dim, args.img_size)
    discriminator = make_discriminator_model(args.img_size)

    print("\n=== Архітектура генератора ===")
    generator.summary()
    print("\n=== Архітектура дискримінатора ===")
    discriminator.summary()

    train(ds, generator, discriminator, args)

if __name__ == "__main__":
    main()
