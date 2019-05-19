import time
from torch.utils.data import DataLoader
from data_loader import custom_dset
import config as cfg
from torchvision import transforms
from model import East
import torch
from torch.optim import lr_scheduler
import loss
import os
import utils
import tensorboardX


def init_tensorboard_writer(store_dir):
    assert os.path.exists(os.path.dirname(store_dir))
    return tensorboardX.SummaryWriter(store_dir)


def fit(train_loader, model, criterion, optimizer, epoch, weight_loss, writer):

    model.train()
    start = time.time()

    for i, (img, img_path, score_map, geo_map, training_mask) in enumerate(train_loader):

        img, score_map, geo_map, training_mask = img.cuda(), score_map.cuda(), geo_map.cuda(), training_mask.cuda()
        f_score, f_geometry = model(img)
        model_loss = criterion(score_map, f_score, geo_map, f_geometry, training_mask)
        total_loss = model_loss + weight_loss(model)

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        end = time.time()
        batch_sum_time = end - start
        per_img_time = 1.0 * batch_sum_time / img.size(0)
        start = end

        steps=epoch*len(train_loader)+i
        writer.add_scalar('model_loss', model_loss.item(), steps)
        writer.add_scalar('total_loss', total_loss.item(), steps)
        writer.add_scalar('per_img_time', per_img_time, steps)

        if i % cfg.print_freq == 0:
            print('EAST <==> TRAIN <==> Epoch: [%d][%d/%d] ,Model Loss %.5f, Total Loss %.5f, Per Img Time %.2f second'
                  % (epoch, i, len(train_loader), model_loss.item(), total_loss.item(), per_img_time))


def main():

    # Prepare for dataset
    print('EAST <==> Prepare <==> DataLoader <==> Begin')
    trainset = custom_dset(transform=transforms.ToTensor())
    train_loader = DataLoader(trainset, batch_size=cfg.train_batch_size_per_gpu * cfg.gpu,
                              shuffle=True, num_workers=cfg.num_workers)
    print('EAST <==> Prepare <==> Batch_size:{} <==> Begin'.format(cfg.train_batch_size_per_gpu * cfg.gpu))
    print('EAST <==> Prepare <==> DataLoader <==> Done')

    # test datalodaer
    # import numpy as np
    # import matplotlib.pyplot as plt
    # for batch_idx, (img, img_path, score_map, geo_map, training_mask) in enumerate(train_loader):
    #     print("batch index:", batch_idx, ",img batch shape", np.shape(geo_map.numpy()))
    #     h1 = img.numpy()[0].transpose(1, 2, 0).astype(np.int64)
    #     h2 = score_map.numpy()[0].transpose(1, 2, 0).astype(np.float32)[:, :, 0]
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(h1)
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(h2, cmap='gray')
    #     plt.show()

    # Model
    print('EAST <==> Prepare <==> Network <==> Begin')
    model = East()
    model = torch.nn.DataParallel(model, device_ids=cfg.gpu_ids)
    criterion = loss.LossFunc().cuda()
    weight_loss = utils.Regularization(model, cfg.l2_weight_decay, p=2).cuda()

    pre_params = list(map(id, model.module.mobilenet.parameters()))
    post_params = filter(lambda p: id(p) not in pre_params, model.module.parameters())
    optimizer = torch.optim.Adam([{'params': model.module.mobilenet.parameters(), 'lr': cfg.pre_lr},
                                  {'params': post_params, 'lr': cfg.lr}])
    # 计算方式 decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.decay_steps, gamma=cfg.decay_rate)
    model.cuda()

    # init or resume，恢复模型
    if cfg.resume and os.path.isfile(cfg.checkpoint):
        start_epoch = utils.Loading_checkpoint(model, optimizer, scheduler)
    else:
        start_epoch = 0

    print('EAST <==> Prepare <==> Network <==> Done')

    tensorboard_writer = init_tensorboard_writer('tensorboards/{}'.format(str(int(time.time()))))

    # train Model
    for epoch in range(start_epoch, cfg.max_epochs):

        scheduler.step()
        fit(train_loader, model, criterion, optimizer, epoch, weight_loss,tensorboard_writer)

        # 保存模型
        if epoch % cfg.save_eval_iteration == 0:
            utils.save_checkpoint(epoch, model, optimizer, scheduler)


if __name__ == "__main__":
    main()
