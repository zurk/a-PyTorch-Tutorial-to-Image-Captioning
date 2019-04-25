import editdistance
import json
import os
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from pprint import pprint

from image_captioning.constants import DIGIT_WORD_MAP_PATH
from image_captioning.create_input_file_for_svhn import output_folder
from image_captioning.datasets import CaptionDataset
from image_captioning.models import Encoder, DecoderWithAttention
from image_captioning.utils import adjust_learning_rate, AverageMeter, clip_gradient, accuracy, \
    save_checkpoint

MAX_NO_IMPROVEMENT = 10
NO_IMPROVEMENT_ADJUST_RATE = 4
TOP_K_ACCURACY = 1
EPOCH_PER_DATASET = 20

# Data parameters
data_folder = str(output_folder)  # folder with data files saved by create_input_files.py
data_name = 'svhn_1_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
encoder_dim = 512  # dimension of encoder network. 512 for ResNet-18 and 2048 for ResNet-101
dropout = 0.5
device = "cuda"  # "cpu"  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 1000  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation metric
batch_size = 512
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_error = float("inf")  # norm edit distance score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none


def main(word_map_file=None):
    """
    Training and validation.
    """

    global best_error, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    if word_map_file is None:
        word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       encoder_dim=encoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        metrics_train = []
        metrics_val = []

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_error = checkpoint['best_error']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

        metrics_train = checkpoint["metrics"]["train"]
        metrics_val = checkpoint["metrics"]["val"]

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', epoch_per_dataset=EPOCH_PER_DATASET,
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL',
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == MAX_NO_IMPROVEMENT:
            break
        if (epochs_since_improvement > 0 and
                epochs_since_improvement % NO_IMPROVEMENT_ADJUST_RATE == 0):
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        metrics_train.append(train(train_loader=train_loader,
                                   encoder=encoder,
                                   decoder=decoder,
                                   criterion=criterion,
                                   encoder_optimizer=encoder_optimizer,
                                   decoder_optimizer=decoder_optimizer,
                                   epoch=epoch))

        # One epoch's validation
        metrics_val.append(validate(val_loader=val_loader,
                                    encoder=encoder,
                                    decoder=decoder,
                                    criterion=criterion))

        recent_error = metrics_val[-1]["error"]
        # Check if there was an improvement
        is_best = recent_error < best_error
        best_error = min(recent_error, best_error)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        metrics = {
            "train": metrics_train,
            "val": metrics_val,
        }
        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, metrics, best_error, is_best)

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    pprint(metrics_train)
    pprint(metrics_val)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    topk_accs = AverageMeter()  # topk accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        topk = accuracy(scores, targets, TOP_K_ACCURACY)
        losses.update(loss.item(), sum(decode_lengths))
        topk_accs.update(topk, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error_val:.5f} ({error_avg:.5f})'.format(
                  epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                  loss=losses, error_val=1-topk_accs.val/100, error_avg=1-topk_accs.avg/100))

    train_loader.dataset.increment_chunk()

    return {
        "losses": losses.avg,
        "error": 1 - topk_accs.avg/100,
    }


def validate(val_loader, encoder, decoder, criterion, print_freq=None):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: metrics
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    topkaccs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating metrics
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            topk = accuracy(scores, targets, TOP_K_ACCURACY)
            topkaccs.update(topk, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if print_freq is not None and i % print_freq == 0:
                # FIXME
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                      'ERROR {error_val:.5f} ({error_avg:.5f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, topk=topkaccs,
                    error_avg=1-topkaccs.avg/100, error_val=1-topkaccs.val/100))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

    # Calculate metrics
    metrics = calc_edit_distance(references, hypotheses)

    print(
        '\n * LOSS - {loss.avg:.5f}, ERROR - {error:.5f}, '
        'norm edit distance - {edit_distance:.5f}\n'.format(
            loss=losses,
            error=1-topkaccs.avg/100,
            edit_distance=metrics["norm_edit"],
        ))

    metrics.update({
        "losses": losses.avg,
        "error": 1 - topkaccs.avg/100,
    })

    return metrics


def calc_edit_distance(references, hypotheses):
    distances = AverageMeter()
    norm_distances = AverageMeter()
    for r, h in zip(references, hypotheses):
        dist = editdistance.eval(r[0], h)
        distances.update(dist)
        norm_distances.update(dist / len(r[0]))

    return {
        "edit": distances.avg,
        "norm_edit": norm_distances.avg,
    }


if __name__ == '__main__':
    main(str(DIGIT_WORD_MAP_PATH))
