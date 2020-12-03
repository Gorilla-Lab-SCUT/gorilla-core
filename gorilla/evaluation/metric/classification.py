# Copyright (c) Gorilla-Lab. All rights reserved.


def accuracy(output, label, topk=(1, ), mode="percentage"):
    r"""
    Computes the precision@k for the specified values of k
    Args:
        output (tensor): The output of the model, the shape is [batch_size, num_classes]
        label (tensor): The label of the input, the shape is [batch_size, 1]
        topk (tuple, optional): The specified list of value k, default just compute the top1_acc
        mode ("percentage" | "number"): Mode for computing result (correct percentage / number of correct samples)
    Return:
        result (list): Each element contain an accuracy value for a specified k
    Example1:
        pred1 = accuracy(output, label)
    Example2:
        pred1, pred5 = accuracy(output, label, topk=(1, 5))
    """
    assert mode in ["percentage", "number"]
    maxk = max(topk)
    batch_size = output.size(0)  # The type of batch_size is int, not tensor
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = float(correct[:k].sum())
        if mode == "percentage":
            result.append(correct_k * 100.0 / batch_size)
        elif mode == "number":
            result.append(correct_k)

    return result


def accuracy_for_each_class(output, label, total_vector, correct_vector):
    r"""
    Computes the precision for each class: n_correct_among_them / n_truly_x
    Args:
        output (tensor): The output of the model, the shape is [batch_size, num_classes]
        label (tensor): The label of the input, the shape is [batch_size, 1]
        topk (tuple, optional): The specified list of value k, default just compute the top1_acc
        mode ("percentage" | "number"): Mode for computing result (correct percentage / number of correct samples)
    Return:
        result (list): Each element contain an accuracy value for a specified k
    Example:
        pred1 = accuracy(output, label)
        pred1, pred5 = accuracy(output, label, topk=(1, 5))
    
    """
    batch_size = label.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        total_vector[int(label[i])] += 1
        correct_vector[int(label[i])] += correct[i]

    return total_vector, correct_vector


def accuracy_for_each_class_other(output, label, predict_vector,
                                  correct_vector):
    r"""
    Computes the precision for each class: n_correct_among_them / n_predict_x

    """
    batch_size = label.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        predict_vector[int(pred[0, i])] += 1
        correct_vector[int(pred[0, i])] += correct[i]

    return predict_vector, correct_vector
