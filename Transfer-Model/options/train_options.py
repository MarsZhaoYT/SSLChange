import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--augmentation', action='store_true', help='Default for no aug')
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--gpu_ids', type=str, default='1')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--num_channels', type=int, default=3)
parser.add_argument('--EF', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--lr_gamma', type=float, default=0.9)
parser.add_argument('--loss_function', type=str, default='hybrid')
parser.add_argument('--dataset_dir', type=str, required=True,
                    default='datasets/LEVIR-CD/', help='dataset root path')
parser.add_argument('--weight_dir', type=str, default='/workspace/Change_Detection/Transfer-Model/outputs/')
parser.add_argument('--log_dir', type=str, default='/workspace/Change_Detection/Transfer-Model/log/')
parser.add_argument('--name', type=str, default='4_9_simsiam_levir_b4_backbone_proj')
parser.add_argument('--pretrained_model', type=str, required=True,
                    default='YOUR_PROJECT/latest_net_SSLChange.pth')
parser.add_argument('--head_type', type=str, default='sslchange_head')
parser.add_argument('--classifier_name', type=str, default='SNUNet', required=True)
parser.add_argument('--release_pretrain', action='store_true', help='Default: Do not release while fine-tuning')




