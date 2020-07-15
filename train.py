import time

import torch
from tqdm import tqdm

import utils
from dataset import CaptchaDataset
from loss import criterion
from net import Net
from predict import pred_to_text

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 60

TRAIN_TXT_FILE = 'dataset/train.txt'
EVAL_TXT_FILE = 'dataset/val.txt'
NUM_CLASSES = 10

PRINT_INTERVAL = 50
DEVICE = 'cuda'

def train_epoch(epoch):
    max_step = len(train_dataloader)
    am = utils.AverageMeter()
    model.train()
    for i, (images, labels) in enumerate(train_dataloader):
        pred = model(images.to(DEVICE))
        loss = criterion(pred, labels.to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        am.update(loss.item())
        if i != 0 and i % PRINT_INTERVAL == 0:
            print('[{}/{}]\tstep: {}/{}\tlr: {:.6f}\tloss: {:.2f}'.format(
                epoch, EPOCHS, i, max_step, scheduler.get_last_lr()[0], am.get_avg_reset()
            ))

@torch.no_grad()
def validate(epoch):
    am = utils.AccMeter()
    model.eval()
    for images, texts in tqdm(eval_dataloader):
        pred = model(images.to(DEVICE))
        pred_texts = pred_to_text(pred)
        am.add(pred_texts, texts)
    acc = am.acc()
    print('[{}/{}]\tacc: {:.2f}%'.format(
        epoch, EPOCHS, acc * 100
    ))
    return acc

model = Net(NUM_CLASSES).to(DEVICE)
# model.load_state_dict(torch.load('weights/model.pt'))

train_dataset = CaptchaDataset(TRAIN_TXT_FILE, NUM_CLASSES)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
eval_dataset = CaptchaDataset(EVAL_TXT_FILE, NUM_CLASSES, training=False)
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45, 55], gamma=0.1)

start = time.time()
for epoch in range(1, EPOCHS+1):
    train_epoch(epoch)
    validate(epoch)
    scheduler.step()
print('{}s'.format(time.time() - start))

torch.save(model.state_dict(), 'weights/model.pt')
