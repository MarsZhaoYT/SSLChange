import torch
import torch
import torch.utils.data
from utils.parser import print_options
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from options import test_options

# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in train.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.

parser = test_options.parser
opt = parser.parse_args()
print_options(opt)

dev = torch.device('cuda:{}'.format(opt.gpu_ids) if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

classifier_name = opt.classifier_name
name = opt.name

eval_epoch = opt.eval_epoch

# path = 'weights/snunet-32.pt'   # the path of the model
backbone_path = 'Transfer-Model/checkpoint/' + classifier_name + '/' + name + '/backbone_epoch_' + str(eval_epoch) + '.pt'
up_path = 'Transfer-Model/checkpoint/' + classifier_name + '/' + name + '/up_epoch_' + str(eval_epoch) + '.pt'
classifier_path = 'Transfer-Model/checkpoint/' + classifier_name + '/' + name + '/classifier_epoch_' + str(eval_epoch) + '.pt'


backbone = torch.load(backbone_path).to(dev)
up_module = torch.load(up_path).to(dev)
classifier = torch.load(classifier_path).to(dev)

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}

backbone.eval()
up_module.eval()
classifier.eval()
# 模型状态转为eval

with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        # 读取一个batch的数据img1和img2
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        # 获取网络预测结果
        feat_A, feat_B = up_module(backbone(batch_img1)), up_module(backbone(batch_img2))    # [4, 3, 256, 256]

        feat_A = torch.cat([batch_img1, feat_A], 1)
        feat_B = torch.cat([batch_img2, feat_B], 1)

        cd_preds = classifier(feat_A, feat_B)
        # cd_preds = classifier(batch_img1, batch_img2)
        _, cd_preds = torch.max(cd_preds, 1)    # 二值化变化图

        # 将label和变化图输入confusion matrix中计算4个指标
        tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                         cd_preds.data.cpu().numpy().flatten(), labels=[0, 1]).ravel()

        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']

# 分别计算Precision, Recall和F1 Score
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)
IoU = tp / (fp + fn + tp)

print('Precision: {}\nRecall: {}\nF1-Score: {}\nmIoU: {}'.format(P, R, F1, IoU))
