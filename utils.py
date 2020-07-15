class AccMeter:

    def __init__(self):
        self.count = 0
        self.pos = 0

    def add(self, preds, labels):
        for p, l in zip(preds, labels):
            self.pos += 1 if p == l else 0
            self.count += 1

    def acc(self):
        if self.count == 0: return float('NaN')
        return self.pos / self.count

    def clear(self):
        self.count = self.pos = 0

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, temp_sum, n=1):
        self.sum += temp_sum
        self.count += n

    def get_avg_reset(self):
        if self.count == 0:
            return 0.
        avg = float(self.sum) / float(self.count)
        self.reset()
        return avg