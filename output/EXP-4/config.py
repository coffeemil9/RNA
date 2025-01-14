class general:
    seed=2025
    aux=False
    aux_targets=None
    split='kf' #['gkf', 'skf', 'full_train']
    n_splits=5

    
class training:
    model_name='cat'
    target_type='target4'
    batch_size=64
    accum_iter=1
    num_epoch=200
    
class optimizer:        
    weight_decay=1e-2
    lr=1e-5
    eps=1e-6
    betas=(0.9, 0.999)

class scheduler:
    scheduler_type='CosineAnnealingLR' 
    class CosineAnnealingLR:
        min_lr=1e-6
        T_max=500

############################
# 分解用コード wandb用
############################

def class_to_dict(cls):
    return {attr: getattr(cls, attr) for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith("__")}

# # Collect all attributes into one class
# class train_config:
#     # General
#     for k, v in class_to_dict(general).items():
#         vars()[k] = v

#     # Training
#     for k, v in class_to_dict(training).items():
#         vars()[k] = v

#     # Model
#     for k, v in class_to_dict(model).items():
#         vars()[k] = v

#     # Optimizer
#     for k, v in class_to_dict(optimizer).items():
#         vars()[k] = v
#     for k, v in class_to_dict(optimizer.swa).items():
#         vars()['swa_' + k] = v

#     # Scheduler
#     for k, v in class_to_dict(scheduler).items():
#         vars()[k] = v
#     for k, v in class_to_dict(scheduler.CosineAnnealingLR).items():
#         vars()['CosineAnnealingLR' + k] = v

#     # Criterion
#     for k, v in class_to_dict(criterion).items():
#         vars()[k] = v
#     for k, v in class_to_dict(criterion.smooth_l1_loss).items():
#         vars()['smooth_l1_loss_' + k] = v
#     for k, v in class_to_dict(criterion.mse_loss).items():
#         vars()['mse_loss_' + k] = v
#     for k, v in class_to_dict(criterion.rmse_loss).items():
#         vars()['rmse_loss_' + k] = v