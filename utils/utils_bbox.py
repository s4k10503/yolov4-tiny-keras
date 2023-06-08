import tensorflow as tf
from keras import backend as K

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    """
    YOLO出力のバウンディングボックスを修正して、実際の画像の形状に合わせる。

    Args:
        box_xy (tensor): バウンディングボックスの中心の(x, y)座標。
        box_wh (tensor): バウンディングボックスの幅と高さ(w, h)。
        input_shape (tensor): モデルへの入力の形状。
        image_shape (tensor): 元画像の形状。
        letterbox_image (bool): 元画像がletterbox形式かどうか。

    Returns:
        boxes (tensor): 修正されたバウンディングボックス。
    """

    # box_xyとbox_whを縦横反転
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    # input_shapeとimage_shapeをbox_yxと同じデータ型に変換
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        # 画像の実際の形状と入力形状との比率を計算し、新しい形状を決定
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))

        # オフセットは画像の有効領域が画像の左上隅からどれだけずれているかを示す
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        # 縮尺に基づいてバウンディングボックスの中心座標と幅・高さを修正
        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)

    # 座標を結合して画像の形状に基づいてスケーリング
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    """
    アンカーボックスを取得し、モデルの出力をデコード。

    Args:
        feats (tensor): モデルからの出力特徴。
        anchors (list): 使用するアンカーボックスのリスト。
        num_classes (int): 分類するクラスの数。
        input_shape (tensor): モデルへの入力の形状。
        calc_loss (bool): 損失を計算するかどうか。

    Returns:
        tuple: calc_lossがTrueの場合、グリッド、特徴、ボックスの座標、ボックスの形状。
               calc_lossがFalseの場合、ボックスの座標、ボックスの形状、ボックスの信頼度、クラス確率。
    """

    num_anchors = len(anchors)

    # 特徴マップの縦と横のサイズを取得
    grid_shape = K.shape(feats)[1:3]

    # それぞれの特徴マップのグリッド点に対する座標を取得
    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid    = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))

    # 入力されたアンカーボックスをリサイズし、各グリッド位置に対して同じ形状のアンカーボックスを作成
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    # ネットワークからの出力を適切な形状に変形
    feats   = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # ボックスの中心座標と幅・高さを計算。出力はシグモイド関数を経て[0,1]に正規化され、その後座標やアンカーサイズに応じてスケールが変更される
    box_xy          = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh          = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

    # ボックスの信頼度とクラス確率を計算。出力はシグモイド関数を経て[0,1]に正規化される
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # ロスを計算する場合は、その他の中間出力とともに、ボックスの中心座標と幅・高さを返す
    # 予測を行う場合は、ボックスの中心座標と幅・高さ、信頼度、クラス確率を返す
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def DecodeBox(outputs, anchors, num_classes, image_shape, input_shape, anchor_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], max_boxes=100, confidence=0.5, nms_iou=0.3, letterbox_image=True):
    """
    モデルからの出力をデコードして、バウンディングボックス、スコア、クラスを返す。

    Args:
        outputs (list): モデルからの出力のリスト。
        anchors (list): 使用するアンカーボックスのリスト。
        num_classes (int): 分類するクラスの数。
        image_shape (tensor): 元画像の形状。
        input_shape (tensor): モデルへの入力の形状。
        anchor_mask (list): アンカーボックスを選択するためのマスク。
        max_boxes (int): 返すバウンディングボックスの最大数。
        confidence (float): バウンディングボックスの信頼度の閾値。
        nms_iou (float): 非最大抑制のIOUの閾値。
        letterbox_image (bool): 元画像がletterbox形式かどうか。

    Returns:
        boxes_out (tensor): バウンディングボックスのリスト。
        scores_out (tensor): スコアのリスト。
        classes_out (tensor): クラスのリスト。
    """

    # アンカーボックスを取得し、バウンディングボックスの座標とサイズ、信頼度、クラス確率を計算
    box_xy = []
    box_wh = []
    box_confidence  = []
    box_class_probs = []
    for i in range(len(outputs)):
        sub_box_xy, sub_box_wh, sub_box_confidence, sub_box_class_probs = get_anchors_and_decode(outputs[i], anchors[anchor_mask[i]], num_classes, input_shape)
        box_xy.append(K.reshape(sub_box_xy, [-1, 2]))
        box_wh.append(K.reshape(sub_box_wh, [-1, 2]))
        box_confidence.append(K.reshape(sub_box_confidence, [-1, 1]))
        box_class_probs.append(K.reshape(sub_box_class_probs, [-1, num_classes]))

    box_xy          = K.concatenate(box_xy, axis=0)
    box_wh          = K.concatenate(box_wh, axis=0)
    box_confidence  = K.concatenate(box_confidence, axis=0)
    box_class_probs = K.concatenate(box_class_probs, axis=0)

    # バウンディングボックスの座標とサイズを実際のピクセル値に変換
    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

    # バウンディングボックスの信頼度とクラス確率を掛け合わせて、最終的なクラススコアを計算
    box_scores  = box_confidence * box_class_probs

    # スコアが指定したconfidence以下のバウンディングボックスをマスク
    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    # クラスごとにバウンディングボックスを選択し、非最大抑制(NMS)を実施
    boxes_out   = []
    scores_out  = []
    classes_out = []
    for c in range(num_classes):
        class_boxes      = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)

    # 最終的なバウンディングボックス、スコア、クラスを結合
    boxes_out      = K.concatenate(boxes_out, axis=0)
    scores_out     = K.concatenate(scores_out, axis=0)
    classes_out    = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out