"""
"""

from copy import deepcopy

from torch_ecg.cfg import CFG
from torch_ecg.model_configs import (  # noqa: F401
    # cnn bankbone
    vgg_block_basic,
    vgg_block_mish,
    vgg_block_swish,
    vgg16,
    vgg16_leadwise,
    resnet_block_basic,
    resnet_bottle_neck_B,
    resnet_bottle_neck_D,
    resnet_block_basic_se,
    resnet_block_basic_gc,
    resnet_bottle_neck_se,
    resnet_bottle_neck_gc,
    resnet_nature_comm,
    resnet_nature_comm_se,
    resnet_nature_comm_gc,
    resnet_nature_comm_bottle_neck,
    resnet_nature_comm_bottle_neck_se,
    resnetN,
    resnetNB,
    resnetNS,
    resnetNBS,
    tresnetF,
    tresnetP,
    tresnetN,
    tresnetS,
    tresnetM,
    multi_scopic_block,
    multi_scopic,
    multi_scopic_leadwise,
    densenet_leadwise,
    xception_leadwise,
    # lstm
    lstm,
    attention,
    # mlp
    linear,
    # attn
    non_local,
    squeeze_excitation,
    global_context,
    # the whole model config
    ECG_CRNN_CONFIG,
    ECG_SEQ_LAB_NET_CONFIG,
    ECG_UNET_VANILLA_CONFIG,
)


__all__ = [
    "ModelArchCfg",
]


# wav2vec2
wav2vec2 = CFG()

wav2vec2.cnn = CFG()
wav2vec2.cnn.name = "wav2vec2_base"
wav2vec2.cnn.wav2vec2_base = CFG()
wav2vec2.cnn.wav2vec2_base.norm_mode = "group_norm"
wav2vec2.cnn.wav2vec2_base.ch_ks_st = (
    # channels, kernel size, stride
    [[512, 10, 5]]
    + [[512, 3, 2]] * 4
    + [[512, 2, 2]] * 2
)
wav2vec2.cnn.wav2vec2_base.bias = False
wav2vec2.cnn.resnet_nature_comm_bottle_neck_se = deepcopy(
    resnet_nature_comm_bottle_neck_se
)
wav2vec2.cnn.tresnetN = deepcopy(tresnetN)

wav2vec2.encoder = CFG()
wav2vec2.encoder.name = "wav2vec2_small"

wav2vec2.encoder.wav2vec2_small = CFG()
wav2vec2.encoder.wav2vec2_small.embed_dim = 3 * 2**7  # 384
wav2vec2.encoder.wav2vec2_small.projection_dropout = 0.1
wav2vec2.encoder.wav2vec2_small.pos_conv_kernel = 64
wav2vec2.encoder.wav2vec2_small.pos_conv_groups = 16
wav2vec2.encoder.wav2vec2_small.num_layers = 4
wav2vec2.encoder.wav2vec2_small.num_heads = 12
wav2vec2.encoder.wav2vec2_small.attention_dropout = 0.1
wav2vec2.encoder.wav2vec2_small.ff_interm_features = 3 * 2**9  # 1536
wav2vec2.encoder.wav2vec2_small.ff_interm_dropout = 0.1
wav2vec2.encoder.wav2vec2_small.dropout = 0.1
wav2vec2.encoder.wav2vec2_small.layer_norm_first = False
wav2vec2.encoder.wav2vec2_small.layer_drop = 0.1

wav2vec2.encoder.wav2vec2_nano = CFG()
wav2vec2.encoder.wav2vec2_nano.embed_dim = 3 * 2**7  # 384
wav2vec2.encoder.wav2vec2_nano.projection_dropout = 0.1
wav2vec2.encoder.wav2vec2_nano.pos_conv_kernel = 64
wav2vec2.encoder.wav2vec2_nano.pos_conv_groups = 16
wav2vec2.encoder.wav2vec2_nano.num_layers = 3
wav2vec2.encoder.wav2vec2_nano.num_heads = 12
wav2vec2.encoder.wav2vec2_nano.attention_dropout = 0.1
wav2vec2.encoder.wav2vec2_nano.ff_interm_features = 3 * 2**8  # 768
wav2vec2.encoder.wav2vec2_nano.ff_interm_dropout = 0.1
wav2vec2.encoder.wav2vec2_nano.dropout = 0.1
wav2vec2.encoder.wav2vec2_nano.layer_norm_first = False
wav2vec2.encoder.wav2vec2_nano.layer_drop = 0.1

# global pooling
# currently is fixed using `AdaptiveMaxPool1d`
wav2vec2.global_pool = "max"  # "avg", "attn"

wav2vec2.clf = CFG()
wav2vec2.clf.out_channels = [
    1024,
    # not including the last linear layer, whose out channels equals n_classes
]
wav2vec2.clf.activation = "mish"
wav2vec2.clf.bias = True
wav2vec2.clf.kernel_initializer = "he_normal"
wav2vec2.clf.dropouts = 0.2


# mostly follow from torch_ecg.torch_ecg.model_configs.ecg_crnn
ModelArchCfg = CFG()


ModelArchCfg.classification = CFG()
ModelArchCfg.classification.crnn = deepcopy(ECG_CRNN_CONFIG)

ModelArchCfg.classification.wav2vec2 = deepcopy(wav2vec2)

ModelArchCfg.classification.outcome_head = CFG()
ModelArchCfg.classification.outcome_head.out_channels = [
    1024,
    # not including the last linear layer, whose out channels equals n_classes
]
ModelArchCfg.classification.outcome_head.activation = "mish"
ModelArchCfg.classification.outcome_head.bias = True
ModelArchCfg.classification.outcome_head.kernel_initializer = "he_normal"
ModelArchCfg.classification.outcome_head.dropouts = 0.2


ModelArchCfg.segmentation = CFG()
ModelArchCfg.segmentation.seq_lab = deepcopy(ECG_SEQ_LAB_NET_CONFIG)
ModelArchCfg.segmentation.seq_lab.reduction = 1
ModelArchCfg.segmentation.seq_lab.recover_length = True
ModelArchCfg.segmentation.unet = deepcopy(ECG_UNET_VANILLA_CONFIG)


_multi_task_segmentation_head = CFG()
_multi_task_segmentation_head.out_channels = [
    512,
    256,
]  # not including the last linear layer
_multi_task_segmentation_head.activation = "mish"
_multi_task_segmentation_head.bias = True
_multi_task_segmentation_head.kernel_initializer = "he_normal"
_multi_task_segmentation_head.dropouts = [0.2, 0.2, 0.0]
_multi_task_segmentation_head.recover_length = True

ModelArchCfg.multi_task = CFG()
ModelArchCfg.multi_task.crnn = deepcopy(ECG_CRNN_CONFIG)
ModelArchCfg.multi_task.crnn.segmentation_head = deepcopy(_multi_task_segmentation_head)

ModelArchCfg.multi_task.wav2vec2 = deepcopy(wav2vec2)
ModelArchCfg.multi_task.segmentation_head = deepcopy(_multi_task_segmentation_head)
