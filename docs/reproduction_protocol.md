# STHF Reproduction Protocol

## Project Goal

We are reproducing the paper:

Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.

The reproduction target is the fixed STHF architecture first. After that, we compare it against an adaptive multi-cutoff ST-HPF variant.

## Main Paper Components

The reproduced fixed STHF model contains:

1. ResNet-50 backbone
2. Fixed Spatial-Temporal High-Pass Filter (ST-HPF)
3. Modality Purifier
4. Shallow Detail Compensation (SDC)
5. Deep Semantic Refinement (DSR)
6. Identity loss
7. Triplet loss
8. Intermediate identity loss
9. Intermediate triplet loss

## Paper Settings to Follow

- Backbone: ResNet-50
- Pretraining: ImageNet
- Input size: 288 × 144
- Sequence length: 6
- Batch size: 16
- Batch composition: 4 IDs × 4 video clips × 2 modalities
- Optimizer: Adam
- Initial learning rate: 3.5e-4
- Scheduler: cosine decay
- Fixed ST-HPF spatial cutoff: fs = 10
- Fixed ST-HPF temporal cutoff: ft = 2
- Loss: ID + Triplet + intermediate ID + intermediate Triplet
- Metrics: Rank-1, Rank-5, Rank-10, mAP

## Dataset Batch Contract

All dataloaders must return:

~~~python
{
    "frames": Tensor,      # [B, T, C, H, W]
    "pids": Tensor,        # [B]
    "camids": Tensor,      # [B]
    "modalities": list,    # length B
    "track_ids": list      # length B
}
~~~

## Model Input Contract

All models receive:

~~~python
frames      # Tensor [B, T, C, H, W]
modalities  # optional list[str]
~~~

## Model Output Contract

All models must return:

~~~python
{
    "features": Tensor,          # [B, D]
    "logits": Tensor,            # [B, num_classes]

    "int_features": Tensor | None,
    "int_logits": Tensor | None,

    "extra": dict
}
~~~

## Baseline Output

The baseline model returns:

~~~python
{
    "features": features,
    "logits": logits,
    "int_features": None,
    "int_logits": None,
    "extra": {
        "model_type": "baseline"
    }
}
~~~

## Fixed STHF Output

The fixed STHF model returns:

~~~python
{
    "features": features,
    "logits": logits,
    "int_features": int_features,
    "int_logits": int_logits,
    "extra": {
        "model_type": "sthf_fixed",
        "sthpf_type": "fixed"
    }
}
~~~

## Adaptive STHF Output

The adaptive STHF model returns:

~~~python
{
    "features": features,
    "logits": logits,
    "int_features": int_features,
    "int_logits": int_logits,
    "extra": {
        "model_type": "sthf_adaptive",
        "sthpf_type": "adaptive",
        "filter_weights": weights
    }
}
~~~

## Loss Contract

The loss builder receives:

~~~python
outputs
pids
~~~

and returns:

~~~python
{
    "total": total_loss,
    "id_loss": id_loss,
    "triplet_loss": triplet_loss,
    "int_id_loss": int_id_loss,
    "int_triplet_loss": int_triplet_loss
}
~~~

## Fair Comparison Rule

For the final comparison, fixed STHF and adaptive STHF must use the same:

- dataset split
- sequence length
- batch size
- backbone
- SDC
- DSR
- losses
- optimizer
- learning rate
- number of epochs
- evaluator

The only changed component should be:

Fixed ST-HPF → Adaptive ST-HPF.
