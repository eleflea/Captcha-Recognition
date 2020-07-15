import argparse

import torch
from PIL import Image
from PIL.ImageDraw import Draw

from dataset import preprocess
from net import Net


def nms(preds, threshold=12):
    orders = torch.sort(preds[:, -1], descending=True)[1]
    keep = []
    while orders.numel() > 0 and len(keep) < 4:
        i = orders[0].item()
        keep.append(i)
        orders = orders[1:]
        xcyc = preds[i, :2]
        others = preds[orders, :2]
        distance = (xcyc - others).pow(2).sum(-1).sqrt()
        mask = distance > threshold
        orders = orders[mask]
    results = preds[keep, :]
    results = results[results[:, 0].sort()[1]]
    return results

def parse_pred(pred):
    bs = pred.shape[0]
    max_cls_prob, max_cls_indexes = pred[..., 2:].max(-1, keepdim=True)
    batch_preds = torch.cat(
        [pred[..., :2], max_cls_indexes.float(), max_cls_prob],
        dim=-1
    ).view(bs, -1, 4)
    batch_results = torch.stack([nms(preds) for preds in batch_preds])
    return batch_results

def convert_text(result):
    batch_asciis = result[:, :, 2].to(torch.int).tolist()
    return [''.join(map(lambda x: chr(48+x), asciis)) for asciis in batch_asciis]

def pred_to_text(pred):
    return convert_text(parse_pred(pred))

@torch.no_grad()
def predict(model, img_path):
    image = Image.open(img_path)
    image = preprocess(image).unsqueeze(0)
    pred = model(image)
    return pred_to_text(pred)[0]

@torch.no_grad()
def draw_detail(model, img_path, save_path):
    image = Image.open(img_path)
    inputs = preprocess(image).unsqueeze(0)
    model.eval()
    pred = model(inputs)
    result = parse_pred(pred).squeeze(0).numpy()
    w, h = image.size
    detail = Image.new('RGB', (w, h * 2), (255, 255, 128))
    detail.paste(image, (0, 0))
    draw = Draw(detail)
    red = (255, 0, 0)
    black = (0, 0, 0)
    for r in result:
        xc, yc, ci, cp = r
        cc = chr(48 + int(ci))
        box = [(xc - 2, yc - 2), (xc + 2, yc + 2)]
        draw.ellipse(box, fill=red)
        draw.line([(xc, yc), (xc, h * 1.5)], fill=red)
        draw.text((xc - 2, h * 1.5), cc, fill=black)
        draw.text((xc - 8, h * 1.5 + 10), '{:.2f}'.format(cp), fill=black)
    detail.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict a image')
    parser.add_argument('--weight', help='path to weights', default='weights/model.pt')
    parser.add_argument('--img', help='path to image')
    parser.add_argument('--detail', default='', help='show detail in output image')
    args = parser.parse_args()
    model = Net(10)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    model.eval()
    if args.detail:
        draw_detail(model, args.img, args.detail)
    else:
        print(predict(model, args.img))
