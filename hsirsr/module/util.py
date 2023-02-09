
def listmap(func, *iterables):
    return list(map(func, *iterables))


def cal_metric(func, output, target):
    if len(output.shape) == 5:
        output = output.squeeze(1)
        target = target.squeeze(1)
    return func(output, target)
