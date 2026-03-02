import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sche
import yaml
from progress.bar import Bar
from model.loss import MultiLoss
from utils.dataloader import PairLoader, get_loader
from model.model import MergedNet
from torch.utils.tensorboard import SummaryWriter
from utils.pair_dataset import SyntheticPairDataset, TransformedPairs, CatPairDataset
from utils.aachen import AachenImages_DB, AachenPairs_StyleTransferDayNight, AachenPairs_OpticalFlow
from utils.web_images import RandomWebImages
from utils.transforms import *

def device_select(x, device):
    if isinstance(x, dict):
        return {k: device_select(v, device) for k, v in x.items()}
    elif isinstance(x, (tuple, list)):
        return [device_select(v, device) for v in x]
    elif isinstance(x, torch.Tensor):
        return x.contiguous().to(device, non_blocking=True)
    else:
        return x
    
# 动态加载数据集
def load_dataset(config):
    data_sources = {}
    for key, source_config in config['data_sources'].items():
        args = source_config['args']
        
        if key == 'web_images':
            web_images = RandomWebImages(source_config['params']['start'], source_config['params']['end'])            
            transforms = [t for t in args['transforms']]
            data_sources[key] = SyntheticPairDataset(
                web_images,
                *transforms
            )
        elif key == 'aachen_db_images':
            transforms = [t for t in args['transforms']]
            aachen_db_images = AachenImages_DB()
            data_sources[key] = SyntheticPairDataset(
                aachen_db_images,
                *transforms
            )
        elif key == 'aachen_style_transfer_pairs':
            transforms = [t for t in args['transforms']]
            aachen_style_transfer_pairs = AachenPairs_StyleTransferDayNight()
            data_sources[key] = TransformedPairs(
                aachen_style_transfer_pairs,
                *transforms
            )
        elif key == 'aachen_flow_pairs':
            data_sources[key] = AachenPairs_OpticalFlow()

        # elif dataset_type == 'PredefinedDataset':
        #     data_sources[key] = PredefinedDataset(args['name'])
    
    return data_sources

# 动态加载数据加载器
def load_loader(config, data_sources):
    data_list = [data_sources[key] for key in config['train_data']]
    combined_data_list = CatPairDataset(*data_list)
    
    loader_config = config['data_loader']
    loader_type = loader_config['type']
    loader_args = loader_config['args']
    
    if loader_type == 'PairLoader':
        scale = loader_args['scale']
        distort = loader_args['distort']
        crop = loader_args['crop']
        
        data_base = PairLoader(
            combined_data_list,
            scale=scale,
            distort=distort,
            crop=crop
        )
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")
    
    return data_base

if __name__ == '__main__':

    with open('./config/train_config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    log_dir = config['tensorboard_log_dir']

    # 初始化 SummaryWriter
    writer = SummaryWriter(log_dir)

    os.makedirs(config['save_path'], exist_ok=True)

    # 加载数据集和数据加载器
    data_sources = load_dataset(config)
    data_base = load_loader(config, data_sources)
    loader = get_loader(data_base, config['iscuda'], config['threads'], config['batch_size'], shuffle=True)
    total_step = len(loader)

    model = MergedNet()

    if len(config['gpu']) > 1 :
        model = torch.nn.DataParallel(model, device_ids=config['gpu'])
        orig_net = model.module
    else:
        orig_net = model

    device = torch.device(f"cuda:{config['gpu'][0]}")
    model.to(device)


    if config['pretrained']:
        checkpoint = torch.load(config['pretrained_path'], lambda a,b:a)
        model.load_pretrained(checkpoint['state_dict'])
    
    # create optimizer
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'], weight_decay=config['weight_decay'])
    schedule_step = sche.MultiStepLR(optimizer, milestones=config['step_size'], gamma=config['gamma'])

    loss = MultiLoss(config).to(device)
    
    print("Let's go!")
    for epoch in range(config['total_epochs']):
        model.train()

        loss.set_epoch(epoch)

        epoch_total_loss = 0.0
        epoch_loss_list = [0.0] * 5

        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        bar = Bar('{:5}-{:5} | epoch{:3}:'.format('PG2SD', 'Train', epoch), max=total_step)

        for iter,inputs in enumerate(loader, start=1):
            inputs = device_select(inputs, device)

            optimizer.zero_grad()
            descriptors_list, repeatability_list, reliability_list = model(imgs=[inputs.pop('img1'),inputs.pop('img2')])
            total_loss, loss_list = loss(descriptors_list, repeatability_list, reliability_list, aflow=inputs['aflow'])
            
            total_loss.backward()

            if torch.isnan(total_loss):
                raise RuntimeError('Loss is NaN')
            
            optimizer.step()

            epoch_total_loss += total_loss.item()
            for i, single_loss in enumerate(loss_list):
                epoch_loss_list[i] += single_loss.item()

            Bar.suffix = '{:4}/{:4} | Loss: {:.4f}, Reli: {:.4f}, COS: {:.4f}, Peak: {:.4f}, PGL: {:.4f}, G2SDL: {:.4f}'.format(
                iter, total_step, epoch_total_loss / iter, epoch_loss_list[0] / iter, epoch_loss_list[1] / iter, epoch_loss_list[2] / iter, epoch_loss_list[3] / iter, epoch_loss_list[4] / iter)
    
            bar.next()
        bar.finish()

        # 计算每个 epoch 的平均损失
        epoch_avg_total_loss = epoch_total_loss / total_step
        epoch_avg_loss_list = [loss / total_step for loss in epoch_loss_list]

        # 记录每个 epoch 的总损失
        writer.add_scalar('Total Loss', epoch_avg_total_loss, epoch)

        # 记录每个 epoch 的各项损失
        loss_names = ['Reliability Loss', 'Cosine Loss', 'Peak Loss', 'PGL Loss', 'G2SDL Loss']
        for i, single_loss in enumerate(epoch_avg_loss_list):
            writer.add_scalar(f'{loss_names[i]}', single_loss, epoch)

        schedule_step.step()

        if len(config['gpu']) > 1:
            torch.save(model.module.state_dict(), config['save_path'] + 'PG2SD.pth'+ '.%d' % (epoch), _use_new_zipfile_serialization=False)
        else:
            torch.save(model.state_dict(), config['save_path'] + 'PG2SD.pth'+ '.%d' % (epoch), _use_new_zipfile_serialization=False)
