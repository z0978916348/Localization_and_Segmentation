from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, square
from skimage.filters import sobel
from skimage.color import gray2rgb
import numpy as np
from tqdm import tqdm

# 除雜訊
def clean_noice(img, threshold=128, middle_threshold=80):
    img = img > threshold

    # 計算投影直方圖
    val_count = list()
    for i in range(img.shape[1]):
        val_count.append(np.sum(img[:, i] != 0))
    matrix = np.zeros_like(img)
    matrix = np.transpose(matrix)
    for i in range(matrix.shape[0]):
        matrix[i, :val_count[i]] = 1
    matrix = np.transpose(matrix)

    # 找中點，去除目標外資訊
    matrix = label(matrix, connectivity=2)
    max_area = 0
    for group in regionprops(matrix):
        if group.label != 0 and group.area > max_area:
            max_area = group.area
            col_min = group.bbox[1]
            col_max = group.bbox[3]
    img[:, :col_min] = 0
    img[:, col_max + 1:] = 0

    # 找中線，去除雜訊
    middle_line = (col_min + col_max) // 2
    matrix = label(img, connectivity=2)
    for group in regionprops(matrix):
        if group.label != 0 and abs(group.centroid[1] - middle_line) > middle_threshold:
            matrix = np.where(matrix == group.label, 0, matrix)
    img = matrix > 0

    return img * 255


def horizontal_cut(img, ratio=0.7, width=10, min_area=1500):
    pixcl_count = list()
    for i in range(img.shape[0]):
        pixcl_count.append(np.sum(img[i, :] > 0))

    edges = list()
    reset = True
    climb = True
    local_hightest = 0
    local_lowest = 0
    for idx in range(len(pixcl_count)):
        if reset:
            local_hightest = 0
            local_lowest = 0
            reset = False
            climb = True

        if climb:
            local_hightest = idx if pixcl_count[idx] > pixcl_count[local_hightest] else local_hightest
            if pixcl_count[idx] < pixcl_count[local_hightest] * ratio:
                local_lowest = idx
                climb = False
        else:
            local_lowest = idx if pixcl_count[idx] < pixcl_count[local_lowest] else local_lowest
            if pixcl_count[idx] * ratio > pixcl_count[local_lowest]:
                reset = True
                edges.append(local_lowest)

    img = closing(img)

    for i in edges:
        img[i: i + width, :] = 0

    labels = label(img, connectivity=1)
    for group in regionprops(labels):
        if group.area < min_area:
            labels = np.where(labels == group.label, 0, labels)
    return (labels > 0) * 255


def opening(img, square_width=5):
    return binary_dilation(binary_erosion(img, square(square_width)), square(square_width))


def closing(img, square_width=15):
    return binary_erosion(binary_dilation(img, square(square_width)), square(square_width))


def dice_coef(target, truth, t_val=255):
    target_val = target == t_val
    truth_val = truth == t_val
    target_obj = np.sum(target_val)
    truth_obj = np.sum(truth_val)
    intersection = np.sum(np.logical_and(target_val, truth_val))
    dice = (2 * intersection) / (target_obj + truth_obj)
    return dice


def dice_coef_each_region(target, truth):
    target_lab = label(target, connectivity=1)
    truth_lab = label(truth, connectivity=1)
    target_vals = sorted(list(np.unique(target_lab)))[1:]
    truth_vals = sorted(list(np.unique(truth_lab)))[1:]

    
    scores = list()
    for _, truth_num in enumerate(tqdm(truth_vals, total=len(truth_vals))):
        score = 0
        lab = 0
        for target_num in target_vals:
            dice = dice_coef(target_lab == target_num, truth_lab == truth_num, t_val=1)
            if dice > score:
                score = dice
                lab = target_num
        scores.append((truth_num, lab, score))
        if lab != 0 and score != 0:
            target_vals.remove(lab)
       
    # (truth_num, target_num, score)
    return scores, np.mean([i[2] for i in scores])


def draw_edge(origin_img, predict_map):
    img = gray2rgb(origin_img)
    edge = sobel(predict_map)
    row, col = np.where(edge > 0)
    for i in range(len(row)):
        img[row[i], col[i], 0] = 0
        img[row[i], col[i], 1] = 0
        img[row[i], col[i], 2] = 255
    return img


if __name__ == '__main__':
    from skimage import io
    from skimage.color import rgb2gray
    import matplotlib.pyplot as plt

    img_label = rgb2gray(io.imread("original_data/f01/image/0003.png"))
    img_target = rgb2gray(io.imread("test//predict//p0003.png"))

    print(img_label.shape)
    print(img_target.shape)

    # img_label = img_label[:, :-4]
    
    img_target = clean_noice(img_target)
    img_target = horizontal_cut(img_target)
    plt.subplot(1, 2, 1)
    plt.imshow(img_target, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(img_label, cmap="gray")
    plt.show()

    mapping, dice = dice_coef_each_region(img_target, img_label)
    print(mapping)
    print(dice)
