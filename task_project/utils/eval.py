import time

import torch

from .metrics import AverageMeter, Result
from .graph import *


def validate(val_loader, model, epoch, device, output_directory='./results', print_freq=300):
    average_meter = AverageMeter()
    model.to(device)
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.to(device), target.to(device)
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50

        rgb = input

        if i == 0:
            img_merge = merge_into_row(rgb, target, pred)
        elif (i < 8*skip) and (i % skip == 0):
            row = merge_into_row(rgb, target, pred)
            img_merge = add_row(img_merge, row)
        elif i == 8*skip:
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            save_image(img_merge, filename)

        if (i+1) % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    return avg, img_merge


def measure_time_torch(n_iter, data, model, device):
    model.to(device)
    model.eval
    data = data.to(device)
    with torch.no_grad():
        start = time.time()
        for _ in range(n_iter):
            _ = model(data)
        
        end = time.time()
    
    return end - start


def measure_time_onnx(n_iter, data, ort_sess):
    start = time.time()
    for _ in range(n_iter):
        _ = ort_sess.run(output_names=['mask'], input_feed={'image': data.numpy()})
    
    end = time.time()
    
    return end - start