'''
Intersection over Union (IoU)
Given two axis-aligned bounding boxes, calculate their intersection over union.
Each bounding box is represented as (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
'''

import numpy as np

def iou(box1, box2):
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union

# Test 1: Partial overlap
print(iou([0, 0, 2, 2], [1, 1, 3, 3]))
# 0.142857 (1 / 7)

# Test 2: Identical boxes
print(iou([0, 0, 2, 2], [0, 0, 2, 2]))
# 1.0

# Test 3: No overlap
print(iou([0, 0, 2, 2], [3, 3, 5, 5]))
# 0.0

# Test 4: One box inside another
print(iou([0, 0, 4, 4], [1, 1, 3, 3]))
# 0.25

# Test 5: Touching boundaries
print(iou([0, 0, 2, 2], [2, 0, 4, 2]))
# 0.0

# Test 6: Degenerate boxes
print(iou([0, 0, 0, 0], [0, 0, 0, 0]))
# 0.0, under our chosen convention

'''
Vectorized IoU
calculate IoU between one bounding box and N bounding boxes simultaneously.
box.shape    # (4,)
boxes.shape  # (N, 4)
ious.shape   # (N,)
'''


def iou_vectorized(box, boxes):
    # box:   (4,)
    # boxes: (N, 4)

    # 1. Intersection coordinates: each (N,)
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])

    # 2. Intersection areas: (N,)
    inter_w = np.maximum(0, ix2 - ix1)
    inter_h = np.maximum(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    # 3. Individual areas
    area1 = (box[2] - box[0]) * (box[3] - box[1])  # scalar
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (N,)

    # 4. Union areas: (N,)
    union = area1 + area2 - inter_area

    # 5. Calculate IoUs
    return np.where(union > 0, inter_area / union, 0.0)

box = np.array([0., 0., 2., 2.])
boxes = np.array([
    [0., 0., 2., 2.],
    [1., 1., 3., 3.],
    [3., 3., 5., 5.],
    [0., 0., 1., 1.]
])

print(iou_vectorized(box, boxes))

'''
Greedy Non-Maximum Suppression (NMS)
Object detection models often predict multiple overlapping bounding boxes for the same object. NMS removes redundant predictions while retaining high-confidence detections.
boxes.shape   # (N, 4)
scores.shape  # (N,)
threshold     # IoU threshold
'''


def nms(boxes, scores, threshold=0.5):
    """
    boxes:  (N, 4)
    scores: (N,)
    threshold: IoU suppression threshold

    Returns:
        indices of retained boxes
    """

    # 1. Sort indices by descending confidence
    order = np.argsort(-scores)

    keep = []

    while len(order) > 0:

        # 2. Select highest-scoring box
        i = order[0]
        keep.append(i)

        # 3. Get remaining indices
        remaining = order[1:]

        # 4. Compute IoU with remaining boxes
        ious = iou_vectorized(boxes[i], boxes[remaining])

        # 5. Keep boxes below or equal to threshold
        order = remaining[ious <= threshold]

    return keep

boxes = np.array([
    [0., 0., 2., 2.],
    [0.2, 0.2, 2.2, 2.2],
    [5., 5., 7., 7.]
])

scores = np.array([0.9, 0.8, 0.7])

print(nms(boxes, scores, threshold=0.5))

boxes = np.array([
    [0., 0., 2., 2.],
    [0., 0., 2., 2.],
    [5., 5., 7., 7.]
])

scores = np.array([0.5, 0.9, 0.7])

print(nms(boxes, scores, threshold=0.5))
# [1, 2]

boxes = np.array([
    [0., 0., 1., 1.],
    [2., 2., 3., 3.],
    [4., 4., 5., 5.]
])

scores = np.array([0.7, 0.9, 0.8])

print(nms(boxes, scores))
# [1, 2, 0]

boxes = np.empty((0, 4))
scores = np.empty(0)

print(nms(boxes, scores))
# []

'''
What if the boxes belong to different object classes?
In that case, NMS should be applied separately for each class to avoid suppressing boxes of different classes.
'''

def class_aware_nms(boxes, scores, labels, threshold=0.5):
    keep = []

    for cls in np.unique(labels):
        indices = np.where(labels == cls)[0]

        selected = nms(
            boxes[indices],
            scores[indices],
            threshold
        )

        keep.extend(indices[selected].tolist())

    return sorted(keep, key=lambda i: scores[i], reverse=True)