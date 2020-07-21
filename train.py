import time

import torch
from tqdm import tqdm

import utils
from dataset import CaptchaDataset
from loss import criterion
from net import Net
from predict import pred_to_text

BATCH_SIZE = 32
# 学习率
LR = 1e-3
# 训练轮数
EPOCHS = 60

# 训练集图片路径
TRAIN_TXT_FILE = 'dataset/train.txt'
# 验证集图片路径
EVAL_TXT_FILE = 'dataset/val.txt'
# 类别数
NUM_CLASSES = 10

# 打印间隔
PRINT_INTERVAL = 50
# 训练使用的设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_epoch(epoch):
    '''
    在训练集上训练一轮
    '''
    # 一轮训练的步数
    max_step = len(train_dataloader)
    # 训练损失平均值的量表
    am = utils.AverageMeter()
    # 将模型切换到训练模式
    model.train()
    # 一次迭代一个 batch size 的图片和标签
    for i, (images, labels) in enumerate(train_dataloader):
        # 得到模型输出
        pred = model(images.to(DEVICE))
        # 计算损失
        loss = criterion(pred, labels.to(DEVICE))
        # 清除优化器的梯度
        optimizer.zero_grad()
        # 反向传播梯度
        loss.backward()
        # 优化器优化参数
        optimizer.step()
        # 更新损失
        am.update(loss.item())
        # 达到输出的步数时，输出信息
        if i != 0 and i % PRINT_INTERVAL == 0:
            print('[{}/{}]\tstep: {}/{}\tlr: {:.6f}\tloss: {:.2f}'.format(
                epoch, EPOCHS, i, max_step, scheduler.get_last_lr()[0], am.get_avg_reset()
            ))

# 此函数不计算梯度
@torch.no_grad()
def validate(epoch):
    '''
    在验证集上评估准确率
    '''
    # 验证集平均准确率的量表
    am = utils.AccMeter()
    # 将模型切换到推理模式
    model.eval()
    for images, texts in tqdm(eval_dataloader):
        # 得到模型的输出
        pred = model(images.to(DEVICE))
        # 经过后处理得到预测结果
        pred_texts = pred_to_text(pred)
        # 更新平均准确率
        am.add(pred_texts, texts)
    acc = am.acc()
    # 打印结果
    print('[{}/{}]\tacc: {:.2f}%'.format(
        epoch, EPOCHS, acc * 100
    ))
    return acc

# 建立网络模型，并移入指定设备
model = Net(NUM_CLASSES).to(DEVICE)
# model.load_state_dict(torch.load('weights/model.pt'))

# 训练集
train_dataset = CaptchaDataset(TRAIN_TXT_FILE, NUM_CLASSES)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
# 验证集
eval_dataset = CaptchaDataset(EVAL_TXT_FILE, NUM_CLASSES, training=False)
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
# 调度器
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45, 55], gamma=0.1)

# 记录开始时间
start = time.time()
# 记录最好的准确率
best_acc = -1
# 训练指定轮数
for epoch in range(1, EPOCHS+1):
    # 训练一轮
    train_epoch(epoch)
    # 验证
    acc = validate(epoch)
    # 如果这一轮的准确率高于之前最好的，保存模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'weights/best.pt')
    # 调度器执行一步
    scheduler.step()

# 输出花费时间
print('{}s'.format(time.time() - start))
