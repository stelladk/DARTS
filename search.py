""" Search cell """
import copy
import os
import torch
import torch.nn as nn
import numpy as np
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from models.augment_cnn import AugmentCNN
from architect import Architect
from visualize import plot
from logger import Logger


config = SearchConfig()

device = torch.device("cuda")

# experiment logger
exp_logger = Logger(experiment_name=config.exp_name, port=config.port, api=config.api, enabled=not config.no_logger)
exp_logger.setup_tracking(file_path=config.log_path)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


# ── checkpoint / resume ──────────────────────────────────────────────
# Each phase writes to a single fixed filename that gets overwritten every
# epoch, so a long run never accumulates more than one checkpoint per phase.
def save_search_checkpoint(path, next_epoch, model, w_optim, alpha_optim, lr_scheduler,
                            best_top1, best_genotype):
    torch.save({
        "phase": "search",
        "epoch": next_epoch,
        "model_state": model.state_dict(),
        "w_optim_state": w_optim.state_dict(),
        "alpha_optim_state": alpha_optim.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
        "best_top1": best_top1,
        "best_genotype": best_genotype,
    }, path)


def load_search_checkpoint(ckpt, model, w_optim, alpha_optim, lr_scheduler):
    model.load_state_dict(ckpt["model_state"])
    w_optim.load_state_dict(ckpt["w_optim_state"])
    alpha_optim.load_state_dict(ckpt["alpha_optim_state"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])


def save_eval_checkpoint(path, next_epoch, genotype, model, optimizer, lr_scheduler,
                          best_top1, best_state_dict):
    torch.save({
        "phase": "eval",
        "epoch": next_epoch,
        "genotype": genotype,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
        "best_top1": best_top1,
        "best_state_dict": best_state_dict,
    }, path)


def load_eval_checkpoint(ckpt, model, optimizer, lr_scheduler):
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])


def main():
    exp_logger.start_run(group="DARTS", run_name=config.run_name)
    for attr, value in sorted(vars(config).items()):
        if attr in ("logger", "api", "exp_name", "run_name", "port", "log_path", "tmpdir"):
            continue
        exp_logger.log_parameter(attr, str(value))
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=config.cutout_length, validation=False,
        no_augment=config.no_augment)

    # A single checkpoint file per phase, overwritten every epoch, so an
    # interrupted run never leaves behind more than one file per phase.
    search_ckpt_path = os.path.join(config.path, "checkpoint.pt")
    eval_ckpt_path = os.path.join(config.path, "eval_checkpoint.pt")

    resume_ckpt = None
    if config.resume is not None:
        resume_path = config.resume
        if os.path.isdir(resume_path):
            # Accept a run directory (e.g. searchs/<name>/) and pick whichever
            # checkpoint is present, preferring the later eval-phase one.
            candidates = [os.path.join(resume_path, "eval_checkpoint.pt"),
                          os.path.join(resume_path, "checkpoint.pt")]
            found = [c for c in candidates if os.path.isfile(c)]
            if not found:
                raise FileNotFoundError(
                    "No checkpoint.pt or eval_checkpoint.pt found in resume directory: {}".format(resume_path))
            resume_path = found[0]
        # weights_only=False: our checkpoints hold a Genotype namedtuple and
        # optimizer/scheduler state, not just tensors. PyTorch >= 2.6 defaults
        # to weights_only=True, which rejects them.
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        logger.info("Resuming from {} (phase={}, epoch={})".format(
            resume_path, resume_ckpt["phase"], resume_ckpt["epoch"]))

    if resume_ckpt is not None and resume_ckpt["phase"] == "eval":
        # Search already finished in the interrupted run; skip straight to
        # resuming the evaluation-training phase.
        best_genotype = resume_ckpt["genotype"]
        logger.info("Skipping search phase; resuming eval phase with genotype: {}".format(best_genotype))
    else:
        net_crit = nn.CrossEntropyLoss().to(device)
        model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                    net_crit, device_ids=config.gpus)
        model = model.to(device)

        # weights optimizer
        w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                                  weight_decay=config.w_weight_decay)
        # alphas optimizer
        alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                       weight_decay=config.alpha_weight_decay)

        # split data to train/validation
        n_train = len(train_data)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config.batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=config.workers,
                                                   pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config.batch_size,
                                                   sampler=valid_sampler,
                                                   num_workers=config.workers,
                                                   pin_memory=True)

        *_, test_data = utils.get_data(
            config.dataset, config.data_path, cutout_length=0, validation=True, no_augment=True)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=config.batch_size,
                                                  shuffle=False,
                                                  num_workers=config.workers,
                                                  pin_memory=True)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            w_optim, config.epochs, eta_min=config.w_lr_min)

        start_epoch = 0
        best_top1 = 0.
        best_genotype = model.genotype()
        if resume_ckpt is not None:
            load_search_checkpoint(resume_ckpt, model, w_optim, alpha_optim, lr_scheduler)
            start_epoch = resume_ckpt["epoch"]
            best_top1 = resume_ckpt["best_top1"]
            best_genotype = resume_ckpt["best_genotype"]
            logger.info("Resumed search phase from epoch {}".format(start_epoch))

        architect = Architect(model, config.w_momentum, config.w_weight_decay)

        # training loop
        for epoch in range(start_epoch, config.epochs):
            lr_scheduler.step()
            lr = lr_scheduler.get_lr()[0]

            model.print_alphas(logger)

            # training
            train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)

            # validation
            cur_step = (epoch+1) * len(train_loader)
            top1 = validate(valid_loader, model, epoch, cur_step)
            validate(test_loader, model, epoch, cur_step, split_name="test")

            # log
            # genotype
            genotype = model.genotype()
            logger.info("genotype = {}".format(genotype))

            # genotype as a image
            plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
            caption = "Epoch {}".format(epoch+1)
            try:
                plot(genotype.normal, plot_path + "-normal", caption)
                plot(genotype.reduce, plot_path + "-reduce", caption)
            except Exception as e:
                logger.warning("Failed to plot genotype (graphviz may not be installed): {}".format(e))

            # save
            if best_top1 < top1:
                best_top1 = top1
                best_genotype = genotype
                is_best = True
            else:
                is_best = False
            utils.save_checkpoint(model, config.path, is_best)
            save_search_checkpoint(search_ckpt_path, epoch+1, model, w_optim, alpha_optim, lr_scheduler,
                                    best_top1, best_genotype)

            print("")

        logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
        logger.info("Best Genotype = {}".format(best_genotype))
        exp_logger.log_pytorch_model(model, f"DARTS_{config.dataset}", x=None, path=config.tmpdir, run_id=False)

        # Free search-phase memory before evaluation
        del model, architect, w_optim, alpha_optim, lr_scheduler
        del train_loader, valid_loader, test_loader, test_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del train_data

    # Evaluation phase: train and evaluate the discovered architecture ===
    logger.info("=" * 60)
    logger.info("Starting evaluation phase: training discovered architecture")
    logger.info("=" * 60)

    eval_resume_ckpt = resume_ckpt if resume_ckpt is not None and resume_ckpt["phase"] == "eval" else None
    evaluate_architecture(best_genotype, input_channels, n_classes,
                           checkpoint_path=eval_ckpt_path, resume_ckpt=eval_resume_ckpt)

    exp_logger.end_run()


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    exp_logger.log_metric('search/lrate', lr, epoch, "search epoch")

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        cur_step += 1

    exp_logger.log_metric('search/train loss', losses.avg, epoch, "search epoch")
    exp_logger.log_metric('search/train accuracy', top1.avg, epoch, "search epoch")
    exp_logger.log_metric('search/train top5', top5.avg, epoch, "search epoch")
    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, epoch, cur_step, split_name="val"):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "{}: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        split_name.capitalize(), epoch+1, config.epochs, step, len(valid_loader)-1,
                        losses=losses, top1=top1, top5=top5))

    exp_logger.log_metric(f'search/{split_name} loss', losses.avg, epoch, "search epoch")
    exp_logger.log_metric(f'search/{split_name} accuracy', top1.avg, epoch, "search epoch")
    exp_logger.log_metric(f'search/{split_name} top5', top5.avg, epoch, "search epoch")

    logger.info("{}: [{:2d}/{}] Final Prec@1 {:.4%}".format(
        split_name.capitalize(), epoch+1, config.epochs, top1.avg))

    return top1.avg


def evaluate_architecture(genotype, input_channels, n_classes, checkpoint_path=None, resume_ckpt=None):
    """Train the discovered architecture from scratch and evaluate on test set."""
    # Load data with same augmentation as search phase (no cutout)
    *_, eval_train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=config.cutout_length,
        validation=False, no_augment=config.no_augment)
    *_, test_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0,
        validation=True, no_augment=True)

    eval_train_loader = torch.utils.data.DataLoader(eval_train_data,
                                                     batch_size=config.batch_size,
                                                     shuffle=True,
                                                     num_workers=config.workers,
                                                     pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True)

    # Build the discrete architecture
    criterion = nn.CrossEntropyLoss().to(device)
    use_aux = config.eval_aux_weight > 0.
    eval_model = AugmentCNN(input_channels, config.eval_init_channels,
                            n_classes, config.eval_layers, use_aux, genotype)
    eval_model = nn.DataParallel(eval_model, device_ids=config.gpus).to(device)

    mb_params = utils.param_size(eval_model)
    logger.info("Eval model size = {:.3f} MB".format(mb_params))
    exp_logger.log_metric("training/nb of parameters", count_parameters(eval_model.module),
                          0, "epoch")

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(eval_model.parameters(), config.eval_lr,
                                momentum=config.w_momentum,
                                weight_decay=config.w_weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.eval_epochs)

    # Training loop
    start_epoch = 0
    best_top1 = 0.
    best_state_dict = None
    if resume_ckpt is not None:
        load_eval_checkpoint(resume_ckpt, eval_model, optimizer, lr_scheduler)
        start_epoch = resume_ckpt["epoch"]
        best_top1 = resume_ckpt["best_top1"]
        best_state_dict = resume_ckpt["best_state_dict"]
        logger.info("Resumed eval phase from epoch {}".format(start_epoch))

    for epoch in range(start_epoch, config.eval_epochs):
        lr_scheduler.step()
        drop_prob = config.eval_drop_path_prob * epoch / config.eval_epochs
        eval_model.module.drop_path_prob(drop_prob)

        # Train one epoch
        eval_train_epoch(eval_train_loader, eval_model, optimizer, criterion, epoch)

        # Validate on test set
        top1 = eval_validate(test_loader, eval_model, criterion, epoch)

        if best_top1 < top1:
            best_top1 = top1
            best_state_dict = copy.deepcopy(eval_model.state_dict())

        if checkpoint_path is not None:
            save_eval_checkpoint(checkpoint_path, epoch+1, genotype, eval_model, optimizer, lr_scheduler,
                                  best_top1, best_state_dict)

    logger.info("Eval: Final best test Prec@1 = {:.4%}".format(best_top1))
    eval_model.load_state_dict(best_state_dict)
    exp_logger.log_pytorch_model(eval_model, f"DARTS_{config.dataset}_eval", x=None, path=config.tmpdir, run_id=False)


def eval_train_epoch(train_loader, model, optimizer, criterion, epoch):
    """Train the evaluation model for one epoch."""
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_lr = optimizer.param_groups[0]['lr']
    exp_logger.log_metric('training/lrate', cur_lr, epoch, "epoch")

    model.train()

    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        if config.eval_aux_weight > 0.:
            loss += config.eval_aux_weight * criterion(aux_logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.eval_grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Eval Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.eval_epochs, step, len(train_loader)-1,
                    losses=losses, top1=top1, top5=top5))

    exp_logger.log_metric('training/train loss', losses.avg, epoch, "epoch")
    exp_logger.log_metric('training/train accuracy', top1.avg, epoch, "epoch")
    logger.info("Eval Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(
        epoch+1, config.eval_epochs, top1.avg))


def eval_validate(test_loader, model, criterion, epoch):
    """Evaluate the evaluation model on the test set."""
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(test_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(test_loader)-1:
                logger.info(
                    "Eval Test: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.eval_epochs, step, len(test_loader)-1,
                        losses=losses, top1=top1, top5=top5))

    exp_logger.log_metric('training/test loss', losses.avg, epoch, "epoch")
    exp_logger.log_metric('training/test accuracy', top1.avg, epoch, "epoch")
    exp_logger.log_metric('training/test top5', top5.avg, epoch, "epoch")

    logger.info("Eval Test: [{:3d}/{}] Final Prec@1 {:.4%}".format(
        epoch+1, config.eval_epochs, top1.avg))

    return top1.avg


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()
