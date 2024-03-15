import torch
import os
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def save_network(network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        file_name = 'trained_models'
        if not os.path.exists(file_name):
            os.makedirs(file_name, exist_ok=True)
        else:
            count=0
            while os.path.exists(file_name):
                count += 1
                file_name = file_name + "_" + str(count)
            os.makedirs(file_name, exist_ok=True)
        save_path = os.path.join(file_name, save_filename)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)