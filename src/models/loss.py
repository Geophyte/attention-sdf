import torch


def _clamp(x, delta):
    return torch.clamp(x, min=-delta, max=delta)


def _sgn(x):
    sgn_x = torch.sign(x)
    sgn_x[sgn_x == 0] = 1
    return sgn_x


def _L(pred_clamped, true_clamped):
    return torch.abs(pred_clamped - true_clamped)


def L_epsilon(pred_clamped, true_clamped, epsilon):
    return torch.max(_L(pred_clamped, true_clamped) - epsilon, torch.tensor(0))


def L1_epsilon(pred_sdf, true_sdf, epsilon, delta):
    pred_clamped = _clamp(pred_sdf, delta)
    true_clamped = _clamp(true_sdf, delta)
    l_epsilon = L_epsilon(pred_clamped, true_clamped, epsilon)
    loss = torch.mean(l_epsilon)
    return loss


def L1_epsilon_lambda(pred_sdf, true_sdf, epsilon, lambdaa, delta):
    pred_clamped = _clamp(pred_sdf, delta)
    true_clamped = _clamp(true_sdf, delta)
    coefficient = 1 + lambdaa * _sgn(true_clamped) * _sgn(true_clamped - pred_clamped)
    l_epsilon = L_epsilon(pred_clamped, true_clamped, epsilon)
    loss = torch.mean(coefficient * l_epsilon)
    return loss


def L1_clamp_loss(pred_sdf, true_sdf, delta):
    pred_clamped = _clamp(pred_sdf, delta)
    true_clamped = _clamp(true_sdf, delta)
    loss = torch.mean(_L(pred_clamped - true_clamped))
    return loss
