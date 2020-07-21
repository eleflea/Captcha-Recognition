class AccMeter:
    '''
    用来计算平均准确率的类
    '''

    def __init__(self):
        # 总共的个数
        self.count = 0
        # 正确的个数
        self.pos = 0

    def add(self, preds, labels):
        # 一组一组比较是否一致
        for p, l in zip(preds, labels):
            self.pos += 1 if p == l else 0
            self.count += 1

    def acc(self):
        # 获得平均准确率
        if self.count == 0: return float('NaN')
        return self.pos / self.count

    def clear(self):
        # 清除记录
        self.count = self.pos = 0

class AverageMeter:
    '''
    记录平均值的类
    '''

    def __init__(self):
        # 总数
        self.sum = 0
        # 元素的个数
        self.count = 0

    def reset(self):
        # 重置
        self.sum = 0
        self.count = 0

    def update(self, temp_sum, n=1):
        # 加入新值
        self.sum += temp_sum
        self.count += n

    def get_avg_reset(self):
        # 获取平均值并重置
        if self.count == 0:
            return 0.
        avg = float(self.sum) / float(self.count)
        self.reset()
        return avg