###############
# Data Config #
###############
CAD_BIT = 6
EXT_SEQ=11 
MAX_BOX=5
MAX_EXT = MAX_BOX*EXT_SEQ+1
SKETCH_R = 1
EXTRUDE_R = 1
BBOX_RANGE = 1
CUBOID_RANGE = 1
MAX_CAD = 200
MAX_CODE = 35
SKETCH_PAD = 5
EXT_PAD = 2
CODE_PAD = 4

################
# Train Config #
################
TOTAL_TRAIN_EPOCH = 350
CAD_TRAIN_PATH = 'data/model/train_deduplicate.pkl'
DECODER_CONFIG = {
    'hidden_dim': 512,
    'embed_dim': 256,  
    'num_layers': 6,   
    'num_heads': 8,
    'dropout_rate': 0.1
}
CODE_CONFIG = {
    'hidden_dim': 512,
    'embed_dim': 256,  
    'num_layers': 6,   
    'num_heads': 8,
    'dropout_rate': 0.1
}
AUG_RANGE = 2

#################
# Sample Config #
#################
code_top_p = 0.99
top_p_sample = 0.9
top_p_eval = 0.99
RANDOM_SAMPLE_TOTAL = 2000
RANDOM_SAMPLE_BS = 32     
RANDOM_EVAL_TOTAL = 15000
RANDOM_EVAL_BS = 1024  
