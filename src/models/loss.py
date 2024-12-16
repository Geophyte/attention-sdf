import torch


def clamp(x, delta):
    return torch.clamp(x, min=-delta, max=delta)


def sgn(x):
    sgn_x = torch.sign(x)
    sgn_x[sgn_x == 0] = 1
    return sgn_x


def _L(pred_clamped, true_clamped):
    return torch.abs(pred_clamped - true_clamped)


def L_epsilon(pred_clamped, true_clamped, epsilon):
    return torch.max(_L(pred_clamped, true_clamped) - epsilon, torch.tensor(0))


def L1_epsilon_lambda(pred_sdf, true_sdf, epsilon, lambdaa, delta):
    pred_clamped = clamp(pred_sdf, delta)
    true_clamped = clamp(true_sdf, delta)
    coefficient = (1 + lambdaa * sgn(true_clamped) * sgn(true_clamped - pred_clamped))
    l_epsilon = L_epsilon(pred_clamped, true_clamped, epsilon)
    loss = torch.mean(coefficient * l_epsilon)
    return loss


def L1_clamp_loss(pred_sdf, true_sdf, delta):
    pred_clamped = clamp(pred_sdf, delta)
    true_clamped = clamp(true_sdf, delta)
    loss = torch.mean(_L(pred_clamped - true_clamped))
    return loss
