import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def convolution_2d(image, filter, pad, step):
    k_size = filter.shape[0]

    width_out = int((image.shape[0] - k_size + 2 * pad) / step + 1)
    height_out = int((image.shape[1] - k_size + 2 * pad) / step + 1)

    output_image = np.zeros((width_out - 2 * pad, height_out - 2 * pad))

    for i in range(image.shape[0] - k_size + 1):
        for j in range(image.shape[1] - k_size + 1):
            patch_from_image = image[i:i+k_size, j:j+k_size]
            output_image[i, j] = np.sum(patch_from_image * filter)

    return output_image


def cnn_layer(image_volume, filter, pad=1, step=1):
    image = np.zeros((image_volume.shape[0] + 2 * pad, image_volume.shape[1] + 2 * pad, image_volume.shape[2]))

    for p in range(image_volume.shape[2]):
        image[:, :, p] = np.pad(image_volume[:, :, p], (pad, pad), mode='constant', constant_values=0)

    k_size = filter.shape[1]
    depth_out = filter.shape[0]
    width_out = int((image_volume.shape[0] - k_size + 2 * pad) / step + 1)
    height_out = int((image_volume.shape[1] - k_size + 2 * pad) / step + 1)

    feature_maps = np.zeros((width_out, height_out, depth_out))  # has to be tuple with numbers

    n_filters = filter.shape[0]

    for i in range(n_filters):
        convolved_image = np.zeros((width_out, height_out))  # has to be tuple with numbers

        for j in range(image.shape[-1]):
            convolved_image += convolution_2d(image[:, :, j], filter[i, :, :, j], pad, step)
        feature_maps[:, :, i] = convolved_image

    return feature_maps


def image_pixels_255(maps):
    r = np.zeros(maps.shape)
    for c in range(r.shape[2]):
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                if maps[i, j, c] <= 255:
                    r[i, j, c] = maps[i, j, c]
                else:
                    r[i, j, c] = 255
    return r


def relu_layer(maps):
    r = np.zeros_like(maps)
    result = np.where(maps > r, maps, r)
    return result


def pooling_layer(maps, size=2, step=2):
    width_out = int((maps.shape[0] - size) / step + 1)
    height_out = int((maps.shape[1] - size) / step + 1)

    pooling_image = np.zeros((width_out, height_out, maps.shape[2]))

    for c in range(maps.shape[2]):
        ii = 0
        for i in range(0, maps.shape[0] - size + 1, step):
            jj = 0
            for j in range(0, maps.shape[1] - size + 1, step):
                patch_from_image = maps[i:i+size, j:j+size, c]
                pooling_image[ii, jj, c] = np.max(patch_from_image)
                jj += 1
            ii += 1

    return pooling_image


input_image = Image.open("eagle_grayscale.jpeg")
image_np = np.array(input_image)
print(image_np.shape)
# print(np.array_equal(image_np[:, :, 0], image_np[:, :, 1]))
# print(np.array_equal(image_np[:, :, 1], image_np[:, :, 2]))

filter_1 = np.random.random_integers(low=-1, high=1, size=(4, 3, 3, image_np.shape[-1]))
print(filter_1.shape)
filter_1 = np.zeros((4, 3, 3, 3))

filter_1[0, :, :, 0] = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
filter_1[0, :, :, 1] = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
filter_1[0, :, :, 2] = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

filter_1[1, :, :, 0] = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
filter_1[1, :, :, 1] = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
filter_1[1, :, :, 2] = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

filter_1[2, :, :, 0] = np.array([[1, -1, 0], [-1, 0, 1], [-1, 0, 1]])
filter_1[2, :, :, 1] = np.array([[1, -1, 0], [-1, 0, 1], [-1, 0, 1]])
filter_1[2, :, :, 2] = np.array([[1, -1, 0], [-1, 0, 1], [-1, 0, 1]])

filter_1[3, :, :, 0] = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
filter_1[3, :, :, 1] = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
filter_1[3, :, :, 2] = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
print(filter_1.shape)

cnn_1 = cnn_layer(image_np, filter_1, pad=1, step=1)
cnn_1 = image_pixels_255(cnn_1)
print(cnn_1.shape)

relu_1 = relu_layer(cnn_1)
print(relu_1.shape)

pooling_1 = pooling_layer(relu_1, size=2, step=2)
print(pooling_1.shape)


filter_2 = np.random.random_integers(low=-1, high=1, size=(4, 3, 3, cnn_1.shape[-1]))
print(filter_2.shape)

cnn_2 = cnn_layer(pooling_1, filter_2, pad=1, step=1)
cnn_2 = image_pixels_255(cnn_2)
print(cnn_2.shape)

relu_2 = relu_layer(cnn_2)
print(relu_2.shape)

pooling_2 = pooling_layer(relu_2, size=2, step=2)
print(pooling_2.shape)


filter_3 = np.random.random_integers(low=-1, high=1, size=(4, 3, 3, cnn_2.shape[-1]))
print(filter_3.shape)

cnn_3 = cnn_layer(pooling_2, filter_3, pad=1, step=1)
cnn_3 = image_pixels_255(cnn_3)
print(cnn_3.shape)

relu_3 = relu_layer(cnn_3)
print(relu_3.shape)

pooling_3 = pooling_layer(relu_3, size=2, step=2)
print(pooling_3.shape)


n_rows = cnn_1.shape[-1]
figure_1, ax = plt.subplots(nrows=n_rows, ncols=9, edgecolor='Black')

ax[0, 0].imshow(cnn_1[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 0].set_axis_off()
ax[0, 0].set_title('CNN #1')
for i in range(1, n_rows):
    ax[i, 0].imshow(cnn_1[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 0].set_axis_off()

ax[0, 1].imshow(relu_1[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 1].set_axis_off()
ax[0, 1].set_title('ReLU #1')
for i in range(1, n_rows):
    ax[i, 1].imshow(relu_1[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 1].set_axis_off()

ax[0, 2].imshow(pooling_1[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 2].set_axis_off()
ax[0, 2].set_title('Pooling #1')
for i in range(1, n_rows):
    ax[i, 2].imshow(pooling_1[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 2].set_axis_off()

ax[0, 3].imshow(cnn_2[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 3].set_axis_off()
ax[0, 3].set_title('CNN #2')
for i in range(1, n_rows):
    ax[i, 3].imshow(cnn_2[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 3].set_axis_off()

ax[0, 4].imshow(relu_2[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 4].set_axis_off()
ax[0, 4].set_title('ReLU #2')
for i in range(1, n_rows):
    ax[i, 4].imshow(relu_2[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 4].set_axis_off()

ax[0, 5].imshow(pooling_2[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 5].set_axis_off()
ax[0, 5].set_title('Pooling #2')
for i in range(1, n_rows):
    ax[i, 5].imshow(pooling_2[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 5].set_axis_off()

ax[0, 6].imshow(cnn_3[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 6].set_axis_off()
ax[0, 6].set_title('CNN #3')
for i in range(1, n_rows):
    ax[i, 6].imshow(cnn_3[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 6].set_axis_off()

ax[0, 7].imshow(relu_3[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 7].set_axis_off()
ax[0, 7].set_title('ReLU #3')
for i in range(1, n_rows):
    ax[i, 7].imshow(relu_3[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 7].set_axis_off()

ax[0, 8].imshow(pooling_3[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 8].set_axis_off()
ax[0, 8].set_title('Pooling #3')
for i in range(1, n_rows):
    ax[i, 8].imshow(pooling_3[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 8].set_axis_off()

figure_1.canvas.set_window_title('CNN --> ReLU --> Pooling')
plt.show()