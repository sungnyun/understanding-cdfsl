import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import resnet18


class RepeatedMultiStepLR(LambdaLR):
    def __init__(self, optimizer, milestones=(400, 600, 800), gamma=0.1, interval=1000, **kwargs):
        self.milestones = milestones
        self.interval = interval
        self.gamma = gamma
        super().__init__(optimizer, self._lambda, **kwargs)

    def _lambda(self, epoch):
        factor = 1
        for milestone in self.milestones:
            if epoch % self.interval >= milestone:
                factor *= self.gamma
        return factor


def main():
    resnet = resnet18()

    optimizer1 = Adam(resnet.parameters(), lr=0.1)
    optimizer2 = Adam(resnet.parameters(), lr=0.1)

    s1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[400, 600, 800], gamma=0.1)
    s2 = RepeatedMultiStepLR(optimizer2, milestones=[400, 600, 800])
    s1_history = []
    s2_history = []

    for i in range(2000):
        # print("Epoch {:04d}: {:.6f} / {:.6f}".format(i, s1.get_last_lr()[0], s2.get_last_lr()[0]))
        s1_history.append(s1.get_last_lr()[0])
        s2_history.append(s2.get_last_lr()[0])
        s1.step()
        s2.step()

    assert (s1_history[:1000] == s2_history[:1000])
    assert (s1_history[:1000] == s2_history[1000:])

    print("Manual test passed!")


if __name__ == "__main__":  # manual unit test
    main()
