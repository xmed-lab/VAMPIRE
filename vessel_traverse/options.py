import argparse

parser = argparse.ArgumentParser(description='training argument values')

def add_training_parser(parser):
    parser.add_argument("-device", type=str, default="2")
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-check_interval", type=int, default=10)
    parser.add_argument("-k_fold", type=str, default=10)
    parser.add_argument("-prompt_positive_num", type=int, default=1)
    parser.add_argument("-prompt_negative_num", type=int, default=1)
    parser.add_argument("-model_type", type=str, default="vit_h")
    parser.add_argument("-is_local", type=bool, default=False)
    parser.add_argument("-remark", type=str, default="OCTA-500")
    
    parser.add_argument("-data_path", type=str, default="/home/lwangdk/Enface_NHYY_Image_Label/Image")
    parser.add_argument("-fold", type=int, default=0)
    parser.add_argument("-num_samples", type=int, default=800)
    parser.add_argument("-batch_size", type=int, default=2)
    parser.add_argument("-accumulation_steps", type=int, default=1)
    parser.add_argument('-n_last_blocks', default=4, type=int)
    parser.add_argument('-num_labels', default=6, type=int)
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-task', default='test', type=str)
    
    parser.add_argument('--freeze', action='store_true', default=False)

def add_cell_parser(parser):
    parser.add_argument("-metrics", type=list, default=["Dice", "Jaccard", "Hausdorff"])

def add_octa500_2d_parser(parser):
    parser.add_argument("-fov", type=str, default="3mm")
    parser.add_argument("-label_type", type=str, default="LargeVessel")
    parser.add_argument("-metrics", type=list, default=["Dice", "Jaccard", "Hausdorff"])
    