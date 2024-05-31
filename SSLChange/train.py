"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # 调用create_dataset进行数据集创建
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # 读取dataset的长度
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # 创建网络模型
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    # 在epoch范围内开始迭代
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        # 每个epoch的开始时间
        epoch_start_time = time.time()  # timer for entire epoch
        # 每个iter加载数据的开始时间
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        # # 每个epoch开始训练前对学习率进行更新
        # model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        # ----------- 每个Epoch的开始训练入口 -------------
        # 在每个Epoch中的循环，次数 i = size / batch_size
        # enumerate()返回的是(下标, 下标对应数据)格式的的打包数据
        for i, data in enumerate(dataset):  # inner loop within one epoch
            # 同一个epoch中每个iter的开始时间
            iter_start_time = time.time()  # timer for computation per iteration
            
            # 每100个iter计算一次当前iter使用的时间
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            # 总共运行iter数 = 上一轮总计数 + batch_size
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # 将数据解包并送入model计算对应的输出和loss
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            # 更新两个方向的梯度
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            # 每400个iter输出一次图像并保存到HTML文件
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # 每100个iter获取一次当前的loss，并输出如下信息(当前epoch, 当前iter数, )
            if total_iters % opt.print_freq == 0 and total_iters % opt.update_cdnet_freq != 0:    # print training losses and save logging information to the disk
                losses = model.get_current_gan_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                
            # 每4个iter，更新一次cdnet，获取cdnet的loss
            elif total_iters % opt.print_freq == 0 and total_iters % opt.update_cdnet_freq == 0:
                losses = model.get_current_full_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        
        # 每5个epoch保存一次模型，存储格式为：["epoch数"_net_"D/G".pth]
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # 每个epoch开始训练前对学习率进行更新
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))