import torch
import torch.nn as nn
import models.sslchang_net as sslchange_net
from utils.parser import print_options
import utils.helpers as helpers
import logging
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support as prfs
import os
from tensorboardX import SummaryWriter
import datetime
import json
from options import train_options
from models.fresunet import FresUNet
from models.siamunet_conc import SiamUnet_conc
from models.siamunet_diff import SiamUnet_diff
from models.SNUNet import SNUNet_ECAM
from models.USSFCNet.USSFCNet import USSFCNet


def load_pretrained_ContrastiveBackbone(pretrained_model_name, pretrained_head_type):

    full_path = 'Transfer-Model/pretrained_models/' + pretrained_model_name
    cdnet_dict = torch.load(full_path)

    cdnet = sslchange_net.ContrastiveNet(head_type=pretrained_head_type)
    cdnet.load_state_dict(cdnet_dict)

    # ----------- 增加特征重建模块 -----------
    down = nn.Sequential(*list(cdnet.backbone.encoder.children())[:3])   # [b, 64, 128, 128]

    downstream_backbone = down
    
    print('---------- Load part of pretained backbone successfully! ----------- ')

    # 冻结预训练backbone模型参数，其余部分参与训练
    for param in downstream_backbone.parameters():
        param.requires_grad = False

    return downstream_backbone


if __name__ == '__main__':

    parser = train_options.parser
    opt = parser.parse_args()
    print_options(opt)

    # CUDA_VISIBLE_DEVICES = opt.gpu_ids
    device = torch.device('cuda:{}'.format(opt.gpu_ids) if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # 根据参数加载backbone路径
    pretrained_backbone = opt.pretrained_model
    
    head_type = opt.head_type
    
    # backbone = load_pretrained_G(pretrained_G, opt.gpu_ids)
    backbone = load_pretrained_ContrastiveBackbone(pretrained_backbone, head_type).to(device)

    up_module = nn.Sequential(nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                              nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, output_padding=0)).to(device)
    
    # 此处替换不同的classifier [SNUNet | FC-EF | FC-Siam-conc | FC-Siam-diff | FCCDN.FCCDN]
    # 根据classifier_name参数，判断选择Classifier结构
    if opt.classifier_name == 'SNUNet':
        classifier = SNUNet_ECAM(in_ch=6).to(device)
    elif opt.classifier_name == 'FC_EF':
        classifier = FresUNet().to(device)
    elif opt.classifier_name == 'FC_Siam_conc':
        classifier = SiamUnet_conc().to(device)
    elif opt.classifier_name == 'FC_Siam_diff':
        classifier = SiamUnet_diff().to(device)
    elif opt.classifier_name == 'USSFCNet':
        classifier = USSFCNet(in_ch=6, out_ch=2).to(device)

    criterion = helpers.get_criterion(opt)
    train_loader, val_loader = helpers.get_loaders(opt)

    # optimizer = torch.optim.AdamW(backbone.parameters(), lr=opt.learning_rate)
    optimizer_classifier = torch.optim.AdamW(filter(lambda p: p.requires_grad, classifier.parameters()), lr=opt.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_classifier, step_size=8, gamma=opt.lr_gamma)

    metadata = {}
    train_metadata = {}
    best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
    logging.info('STARTING training')
    total_step = -1

    logging.info('Using backbone: ' + opt.classifier_name)
    logging.info('Start experiment: ' + opt.name)


    for epoch in range(opt.epochs):
        
        name = opt.name
        classifier_name = opt.classifier_name

        if not os.path.exists('Transfer-Model/checkpoint/' + classifier_name + '/' + name):
            os.makedirs('Transfer-Model/checkpoint/' + classifier_name + '/' + name)

        with open('Transfer-Model/checkpoint/' + classifier_name + '/' + name + '/train_option.json', 'w') as fout:
            for k in opt.__dict__:
                train_metadata[k] = opt.__dict__[k]
                
            json.dump(train_metadata, fout)

        # 初始化训练和验证指标，返回一个字典，包含：loss/corrects/precisions/recall/f1_score/lr
        train_metrics = helpers.initialize_metrics()
        val_metrics = helpers.initialize_metrics()

        """
        Begin Training
        """
        backbone.train()   # 模型状态置为train
        up_module.train()
        classifier.train()

        logging.info('SET model mode to train!')
        batch_iter = 0
        tbar = tqdm(train_loader)

        for batch_img1, batch_img2, labels in tbar:
            tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
            batch_iter = batch_iter + opt.batch_size    # 累加batch_size到 batch_iter计数器
            total_step += 1
            # Set variables for training
            # 每个batch的数据转换格式
            batch_img1 = batch_img1.float().to(device)    # [4, 3, 256, 256]
            batch_img2 = batch_img2.float().to(device)     # [4, 3, 256, 256]
            labels = labels.long().to(device)     # [4, 256, 256]

            # Zero the gradient
            optimizer_classifier.zero_grad()

            # Get model predictions, calculate loss, backprop
            # 先用自监督预训练Backbone提取特征
            feat_A, feat_B = up_module(backbone(batch_img1)), up_module(backbone(batch_img2))    # [4, 3, 256, 256]

            feat_A = torch.cat([batch_img1, feat_A], 1).to(device)   # [4, 6, 256, 256]
            feat_B = torch.cat([batch_img2, feat_B], 1).to(device)   # [4, 6, 256, 256]

            # 再用分类器进行变化检测
            cd_preds = classifier(feat_A, feat_B).to(device)   # [4, 3, 256, 256]

            # 实例化每个batch损失函数值
            cd_loss = criterion(cd_preds, labels).to(device)
            loss = cd_loss
            loss.requires_grad_(True)
            loss.backward()     # 进行反向传播

            # optimizer_backbone.step()
            optimizer_classifier.step()


            # 获取预测概率
            # torch.max()的返回值是两个张量，values和indices，此处只需要最大值对应的channel，1代表变化区域，0代表未变化
            _, cd_preds = torch.max(cd_preds, 1)    # 获取最大值对应的通道  [4, 256, 256]


            # 1. 计算正确率（预测值和label值相同的百分比）
            # torch.true_divide()：除法，返回浮点数不做整数处理
            # 除法分子：预测值和label相同的像素数量
            # 除法分母：batch_size * 每个label图像像素个数
            cd_corrects = (100 * torch.true_divide((cd_preds.squeeze().byte() == labels.squeeze().byte()).sum(),
                                                (labels.size()[0] * (opt.patch_size**2))))

            # 调用sklearn计算 precision/recall/f1-score
            cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                                cd_preds.data.cpu().numpy().flatten(),
                                average='macro',
                                pos_label=1)

            # 调用set_metrics更新评价指标字典中的值
            train_metrics = helpers.set_metrics(train_metrics,
                                        cd_loss,
                                        cd_corrects,
                                        cd_train_report,
                                        scheduler.get_last_lr())

            # log the batch mean metrics
            # 对于评价指标中每一项的所有值，求平均后存入mean_train_metrics
            mean_train_metrics = helpers.get_mean_metrics(train_metrics)

            for k, v in mean_train_metrics.items():
                # 在tensorboard中加入每一项评价指标的图表，格式例如(cd_losses/train, 8.0845, 10000)
                writer.add_scalars(str(k), {'train': v}, total_step)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        scheduler.step()    # 更新学习率
        logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

        """
        Begin Validation
        """
        backbone.eval()    # 进入验证状态
        classifier.eval()
        with torch.no_grad():
            for batch_img1, batch_img2, labels in val_loader:
                # Set variables for training
                batch_img1 = batch_img1.float().to(device)
                batch_img2 = batch_img2.float().to(device)
                labels = labels.long().to(device)

                # Get predictions and calculate loss
                feat_A, feat_B = up_module(backbone(batch_img1)), up_module(backbone(batch_img2))    # [4, 3, 256, 256]

                feat_A = torch.cat([batch_img1, feat_A], 1).to(device)
                feat_B = torch.cat([batch_img2, feat_B], 1).to(device)

                cd_preds = classifier(feat_A, feat_B).to(device)     # 获取验证集当前batch的预测输出

                cd_loss = criterion(cd_preds, labels).to(device)       # 验证集当前batch的loss

                # cd_preds = cd_preds[-1]
                _, cd_preds = torch.max(cd_preds, 1)

                # 该部分计算并更新评价指标体系与train中的相同
                # Calculate and log other batch metrics
                cd_corrects = (100 * torch.true_divide((cd_preds.squeeze().byte() == labels.squeeze().byte()).sum(),
                                                    (labels.size()[0] * (opt.patch_size**2))))

                cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                    cd_preds.data.cpu().numpy().flatten(),
                                    average='macro',
                                    pos_label=1)

                val_metrics = helpers.set_metrics(val_metrics,
                                        cd_loss,
                                        cd_corrects,
                                        cd_val_report,
                                        scheduler.get_last_lr())

                # log the batch mean metrics
                mean_val_metrics = helpers.get_mean_metrics(val_metrics)

                for k, v in mean_train_metrics.items():
                    writer.add_scalars(str(k), {'val': v}, total_step)

                # clear batch variables from memory
                del batch_img1, batch_img2, labels

            logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

            """
            Store the weights of good epochs based on validation results
            """
            # 如果在验证集val上某个epoch的平均precision/recall/f1_scores比最佳指标要好，则对best_metrics进行更新
            if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
                    or
                    (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
                    or
                    (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):

                # Insert training and epoch information to metadata dictionary
                logging.info('updata the model')

                # 在metadata中增加一个validation_metrics的dict
                metadata['validation_metrics'] = mean_val_metrics

                # Save model and log
                if not os.path.exists('Transfer-Model/tmp'):
                    os.makedirs('Transfer-Model/tmp')

                
                # 更新json文件
                with open('Transfer-Model/checkpoint/' + classifier_name + '/' + name + '/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                    json.dump(metadata, fout)

                # 保存模型pt文件
                torch.save(backbone, 'Transfer-Model/checkpoint/' + classifier_name + '/' + name + '/backbone_epoch_'+str(epoch)+'.pt')
                torch.save(up_module, 'Transfer-Model/checkpoint/' + classifier_name + '/' + name + '/up_epoch_'+str(epoch)+'.pt')
                torch.save(classifier, 'Transfer-Model/checkpoint/' + classifier_name + '/' + name + '/classifier_epoch_'+str(epoch)+'.pt')

                # comet.log_asset(upload_metadata_file_path)
                best_metrics = mean_val_metrics     # 将最佳指标设为更新后的平均指标

            print('An epoch finished.')
    writer.close()  # close tensor board
    print('Done!')

