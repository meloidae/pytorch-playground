import torch
import torch.nn.functional as F


def label_smoothed_cross_entropy(model_outputs,
                                 tgt_labels,
                                 mask=None,
                                 smoothing_eps=0.1,
                                 reduce=True):
    # model_outputs: [batch_size, length, vocab]
    # tgt_labels: [batch_size, length]
    # mask: [batch_size, length]

    n_labels = model_outputs.size(-1)
    model_outputs = model_outputs.view(-1, n_labels)  # [batch_size * length, vocab]
    tgt_labels = tgt_labels.view(-1, 1)  # [batch_size * length, 1]

    # Index by mask
    if mask is not None:
        mask = mask.view(-1)  # [batch_size * length]
        model_outputs = model_outputs[mask]
        tgt_labels = tgt_labels[mask]

    # Calculate loss
    log_probs = F.log_softmax(model_outputs, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=tgt_labels)  # one_hot * -log_prob at target index
    smooth_loss = -log_probs.sum(dim=-1, keepdim=True)  # sum of -lob_prob

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    # Mixture of one_hot and uniform dist
    # (sum of -lob_prob) / n_labels = sum of -log_prob * uniform_dist
    loss = (1 - smoothing_eps) * nll_loss + (smoothing_eps / n_labels) * smooth_loss
    return loss


def main():
    batch_size = 2
    length = 3
    vocab_size = 5

    model_outputs = torch.rand(size=(batch_size, length, vocab_size))
    tgt_labels = [[2, 4, 0], [1, 3, 0]]
    tgt_labels = torch.LongTensor(tgt_labels)
    mask = tgt_labels != 0
    smoothing_eps = 0.1
    reduce = True

    loss = label_smoothed_cross_entropy(model_outputs, tgt_labels, mask, smoothing_eps, reduce)
    print(loss.item())


if __name__ == '__main__':
    main()
