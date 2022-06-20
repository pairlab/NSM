from yacs.config import CfgNode as CN

# Miscellaneous configs
_C = CN()

# Experiment related
_C.exp = CN()
_C.exp.name           = ''
_C.exp.checkpoint_dir = ''
_C.exp.weight_file    = ''
_C.exp.gpus           = [0, ]
_C.exp.num_workers    = 8
_C.exp.batch_size     = 1
_C.exp.num_epochs     = 1000

# Model related
_C.model = CN()
_C.model.encoder     = 'dgcnn'
_C.model.pc_feat_dim = 512
_C.model.transformer_feat_dim = 1024
_C.model.num_heads   = 4
_C.model.num_blocks  = 1

# Data related
_C.data = CN()
_C.data.root_dir       = ''
_C.data.train_csv_file = ''
_C.data.test_csv_file  = ''
_C.data.num_pc_points  = 1024

# Optimizer related
_C.optimizer = CN()
_C.optimizer.lr           = 1e-3
_C.optimizer.lr_decay     = 0.7
_C.optimizer.decay_step   = 2e4
_C.optimizer.weight_decay = 1e-6
_C.optimizer.lr_clip      = 1e-5

def get_cfg_defaults():
    return _C.clone()
