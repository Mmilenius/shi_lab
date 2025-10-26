import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
import imageio
import glob

print(f"TensorFlow версія: {tf.__version__}")

# Перевірка наявності GPU
if not tf.config.list_physical_devices('GPU'):
    print("ПОПЕРЕДЖЕННЯ: GPU не знайдено! Навчання буде на CPU.")
else:
    print("GPU знайдено, навчання буде прискорено.")

# --- 1. Налаштування гіперпараметрів ---
BUFFER_SIZE = 30000
BATCH_SIZE = 256
IMAGE_SIZE = 32
NOISE_DIM = 100
EPOCHS = 50
NUM_CLASSES = 43
num_examples_to_generate = NUM_CLASSES  # Згенеруємо по 1 прикладу на клас

# --- 2. Завантаження та підготовка датасету ---
def load_gtsrb_data(data_dir):
    train_file_path = os.path.join(data_dir, 'train.p')
    print(f"Завантаження даних з pickle-файлу: {train_file_path}")
    try:
        with open(train_file_path, 'rb') as f:
            train_data = pickle.load(f)

        if 'features' in train_data and 'labels' in train_data:
            images = train_data['features']
            labels = train_data['labels']
        else:
            print("ПОМИЛКА: Ключі 'features' або 'labels' не знайдено.")
            return np.array([]), np.array([])

    except FileNotFoundError:
        print(f"ПОМИЛКА: Файл 'train.p' не знайдено.")
        return np.array([]), np.array([])

    images = images.astype('float32')
    images = (images - 127.5) / 127.5  # Нормалізація до [-1, 1]
    labels = labels.astype('int32')

    print(f"Завантажено {len(images)} зображень.")
    print(f"Форма масиву зображень: {images.shape}")

    if images.shape[1] != IMAGE_SIZE or images.shape[2] != IMAGE_SIZE:
        print(f"ПОПЕРЕДЖЕННЯ: Розмір зображень {images.shape[1:3]} не збігається з IMAGE_SIZE={IMAGE_SIZE}")

    return images, labels


print("Завантаження датасету 'german-traffic-sign-dataset'...")
data_directory = r"D:\workspace\shi_lab\datasets"  # <--- ВКАЖІТЬ ВАШ ШЛЯХ
print(f"Використання локальних даних з: {data_directory}")
train_images, train_labels = load_gtsrb_data(data_directory)

if train_images.size == 0:
    print("Не вдалося завантажити зображення. Вихід.")
    exit()

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

print("Датасет успішно створено.")

# --- 3. Модель Генератора (CGAN) ---
def make_generator_model(num_classes):
    noise_input = layers.Input(shape=(NOISE_DIM,), name="noise_input")
    label_input = layers.Input(shape=(1,), name="label_input")

    label_embedding = layers.Embedding(num_classes, 50)(label_input)
    label_embedding = layers.Flatten()(label_embedding)

    merged_input = layers.Concatenate()([noise_input, label_embedding])

    x = layers.Dense(4 * 4 * 1024, use_bias=False)(merged_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Reshape((4, 4, 1024))(x)

    x = layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    output_image = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
                                          padding='same', use_bias=False, activation='tanh')(x)

    model = keras.Model(inputs=[noise_input, label_input], outputs=output_image)
    return model


generator = make_generator_model(NUM_CLASSES)
print("\n--- Архітектура Генератора ---")
generator.summary()

# --- 4. Модель Дискримінатора (CGAN) ---
def make_discriminator_model(num_classes):
    image_input = layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name="image_input")
    label_input = layers.Input(shape=(1,), name="label_input")

    label_embedding = layers.Embedding(num_classes, IMAGE_SIZE * IMAGE_SIZE)(label_input)
    label_embedding = layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 1))(label_embedding)

    merged_input = layers.Concatenate()([image_input, label_embedding])

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(merged_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output_logit = layers.Dense(1)(x)

    model = keras.Model(inputs=[image_input, label_input], outputs=output_logit)
    return model


discriminator = make_discriminator_model(NUM_CLASSES)
print("\n--- Архітектура Дискримінатора ---")
discriminator.summary()

# --- 5. Функції втрат та оптимізатори ---
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.fill(real_output.shape, 0.9), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# --- 6. Крок навчання ---
@tf.function
def train_step(data):
    images, labels = data
    labels = tf.cast(labels, 'int32')

    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True)

        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# --- 7. Візуалізація ---
output_dir = 'gan_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

seed_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM])
seed_labels = tf.constant(np.arange(num_examples_to_generate) % NUM_CLASSES, dtype='int32')

def generate_and_save_images(model, epoch, test_inputs):
    test_noise, test_labels = test_inputs
    predictions = model([test_noise, test_labels], training=False)

    fig = plt.figure(figsize=(10, 10))
    for i in range(predictions.shape[0]):
        plt.subplot(7, 7, i + 1)
        plt.imshow((predictions[i, :, :, :] * 0.5) + 0.5)
        plt.title(f"Клас: {test_labels[i].numpy()}")
        plt.axis('off')

    plt.savefig(os.path.join(output_dir, f'image_at_epoch_{epoch:04d}.png'))
    plt.close(fig)

# --- 8. Цикл навчання ---
def train(dataset, epochs):
    print("\n--- Початок Навчання GAN ---")
    generate_and_save_images(generator, 0, (seed_noise, seed_labels))

    gen_loss_history = []
    disc_loss_history = []

    for epoch in range(epochs):
        start = time.time()

        total_gen_loss = 0
        total_disc_loss = 0
        batch_count = 0

        for image_batch, label_batch in dataset:
            gen_loss, disc_loss = train_step((image_batch, label_batch))
            total_gen_loss += gen_loss
            total_disc_loss += disc_loss
            batch_count += 1

        avg_gen_loss = total_gen_loss / batch_count
        avg_disc_loss = total_disc_loss / batch_count

        gen_loss_history.append(avg_gen_loss.numpy())
        disc_loss_history.append(avg_disc_loss.numpy())

        generate_and_save_images(generator, epoch + 1, (seed_noise, seed_labels))

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"Saved checkpoint for epoch {epoch + 1}")

        print(f'Time for epoch {epoch + 1} is {time.time() - start:.2f} sec')
        print(f'Avg Gen Loss: {avg_gen_loss}, Avg Disc Loss: {avg_disc_loss}')

    generate_and_save_images(generator, epochs, (seed_noise, seed_labels))
    print("Навчання завершено.")

    return gen_loss_history, disc_loss_history

# --- 9. Візуальний аналіз результатів ---
gen_loss_history, disc_loss_history = train(train_dataset, EPOCHS)

print(f"Навчання завершено. Згенеровані зображення збережено в директорії '{output_dir}'.")

print("Побудова графіків втрат генератора та дискримінатора...")
plt.figure(figsize=(10, 5))
plt.plot(gen_loss_history, label='Втрати Генератора (Gen Loss)')
plt.plot(disc_loss_history, label='Втрати Дискримінатора (Disc Loss)')
plt.xlabel('Епохи')
plt.ylabel('Втрати (Loss)')
plt.legend()
plt.title('Історія Втрат (Loss) GAN під час Навчання')
loss_plot_path = os.path.join(output_dir, 'gan_loss_history.png')
plt.savefig(loss_plot_path)
plt.show()
print(f"Графік втрат збережено у: {loss_plot_path}")

print("Створення GIF-анімації процесу генерації...")
gif_path = os.path.join(output_dir, 'gan_training_progress.gif')
image_files = sorted(glob.glob(os.path.join(output_dir, 'image_at_epoch_*.png')))

if image_files:
    image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f.split('epoch_')[-1]))))
    with imageio.get_writer(gif_path, mode='I', duration=0.5, loop=0) as writer:
        for filename in image_files:
            try:
                image = imageio.v2.imread(filename)
                writer.append_data(image)
            except Exception as e:
                print(f"Помилка при читанні файлу {filename}: {e}")

    print(f"GIF-анімацію збережено у: {gif_path}")
else:
    print("Не знайдено зображень для створення GIF.")

print("\n--- Візуальний аналіз завершено ---")
