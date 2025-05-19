import os
import yaml
import time
import torch
import logging
import argparse
import shutil
import warnings
import humanfriendly
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from nets.intensity_extractor.model_enhance import RankModel
from os import makedirs
from pathlib import Path
from os.path import join
from dataset import Dataset
from utils.tools import to_device
from train.reporter import Reporter
from contextlib import contextmanager
from torch.utils.data import DataLoader
from distutils.version import LooseVersion
from utils.model_wJCU import get_model, get_parser, disc_configure_optimizers
from torch.utils.tensorboard import SummaryWriter
from torch_utils.recursive_op import recursive_average
from schedulers.abs_scheduler import AbsBatchStepScheduler
from torch_utils.add_gradient_noise import add_gradient_noise
from torch_utils.set_all_random_seed import set_all_random_seed
from main_funcs.average_nbest_models import average_nbest_models

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

try:
    import fairscale
except ImportError:
    fairscale = None

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
    from torch.cuda.amp import GradScaler
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


    GradScaler = None


def main(args, configs):
    args.distributed = args.multiprocessing_distributed

    if args.distributed:
        args.world_size = args.gpu_num
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=args.gpu_num, args=(args, configs))
    else:
        # Simply call main_worker function
        main_worker(args.gpu_num, args, configs)


def main_worker(gpu, args, configs):
    print("main_worker")
    preprocess_config, model_config, option_config = configs

    # if gpu == 1:
    #     gpu = 3
    print(gpu)
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            rank = args.gpu
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            # args.rank = gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=rank)

    # create model, optimizer, scheduler
    reporter = Reporter()
    parser = get_parser()
    cmd_list = list()

    for key in option_config.keys():
        cmd_list.append("--{}".format(key))
        cmd_list.append("{}".format(option_config[key]))

    train_args = parser.parse_args(cmd_list)

    if isinstance(train_args.keep_nbest_models, int):
        keep_nbest_models = [train_args.keep_nbest_models]
    else:
        if len(train_args.keep_nbest_models) == 0:
            logging.warning("No keep_nbest_models is given. Change to [1]")
            train_args.keep_nbest_models = [1]
        keep_nbest_models = train_args.keep_nbest_models

    if args.resume:
        model, optimizers, schedulers, disc, advloss, disc_optimizers, disc_schedulers = get_model(configs, args, reporter, device, train=True)
    else:
        model, optimizers, schedulers, disc, advloss = get_model(configs, args,reporter, device, train=True)
        disc_optimizers, disc_schedulers = disc_configure_optimizers(disc)

    total_params = 0
    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        shape_str = str(tuple(param.shape))
        logging.info(
            f"[PARAM] {name:60s} | shape={shape_str:20s} | "
            f"dtype={param.dtype!s:8s} | requires_grad={param.requires_grad}"
        )

    for name, param in disc.named_parameters():
        num = param.numel()
        total_params += num
        shape_str = str(tuple(param.shape))
        logging.info(
            f"[PARAM] {name:60s} | shape={shape_str:20s} | "
            f"dtype={param.dtype!s:8s} | requires_grad={param.requires_grad}"
        )
    logging.info(f"Total parameters: {total_params:,}")

    cudnn.benchmark = True

    # batch_size = int(args.batch_size / args.gpu_num)
    batch_size = 32

    train_dataset = Dataset(
        "train.txt", preprocess_config, batch_size, sort=True,
    )
    valid_dataset = Dataset(
        "val.txt", preprocess_config, batch_size, sort=False,
    )

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False, drop_last=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # shuffle=(train_sampler is None),
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        # sampler=train_sampler,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
        pin_memory=True,
        # sampler=valid_sampler,
        drop_last=True,
    )

    start_epoch = reporter.get_epoch() + 1

    train_log_path = join(args.log_path, "train")
    val_log_path = join(args.log_path, "val")
    output_dir = args.result_path
    makedirs(train_log_path, exist_ok=True)
    makedirs(val_log_path, exist_ok=True)
    makedirs(output_dir, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    valid_logger = SummaryWriter(val_log_path)

    # model, loss, tools, dataset backup
    shutil.copy('./tts/fastspeech2/fastspeech2.py', join(args.log_path, 'fastspeech2.py'))
    shutil.copy('./tts/fastspeech2/loss.py', join(args.log_path, 'loss.py'))
    shutil.copy('./utils/tools.py', join(args.log_path, 'tools.py'))
    shutil.copy('./dataset.py', join(args.log_path, 'dataset.py'))

    start_time = time.perf_counter()

    for iepoch in range(start_epoch, args.max_epoch + 1):
        if iepoch != start_epoch:
            logging.info(
                "{}/{}epoch started. Estimated time to finish: {}".format(
                    iepoch,
                    args.max_epoch,
                    humanfriendly.format_timespan(
                        (time.perf_counter() - start_time)
                        / (iepoch - start_epoch)
                        * (args.max_epoch - iepoch + 1)
                    ),
                )
            )
        else:
            logging.info(f"{iepoch}/{args.max_epoch}epoch started")
        set_all_random_seed(train_args.seed + iepoch)

        reporter.set_epoch(iepoch)
        args.word_size = 1

        with reporter.observe("train") as sub_reporter:
            all_steps_are_invalid = train_one_epoch(

                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                iterator=train_loader,
                reporter=sub_reporter,
                summary_writer=train_logger,
                options=train_args,
                distributed=args.distributed,
                ngpu=1,
                epoch=iepoch,
                disc=disc,
                advloss=advloss,
                disc_optimizers=disc_optimizers,
                disc_schedulerss=disc_schedulers
            )

        with reporter.observe("valid") as sub_reporter:
            validate_one_epoch(
                model=model,
                iterator=valid_loader,
                reporter=sub_reporter,
                options=train_args,
                distributed=args.distributed,
                ngpu=1,
                disc=disc,
                advloss=advloss
            )

        # if args.multiprocessing_distributed and gpu == 0:
        # 3. Report the results
        logging.info(reporter.log_message())
        if train_args.use_matplotlib:
            reporter.matplotlib_plot(Path(join(output_dir, "images")))
        if train_logger is not None:
            reporter.tensorboard_add_scalar(train_logger, key1="train")
            reporter.tensorboard_add_scalar(valid_logger, key1="valid")

        # 4. Save/Update the checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "discriminator": disc.state_dict(),
                "reporter": reporter.state_dict(),
                "optimizers": [o.state_dict() for o in optimizers],
                "disc_optimizers": disc_optimizers.state_dict(),
                "schedulers": [
                    s.state_dict() if s is not None else None
                    for s in schedulers
                ],
                "disc_schedulers": disc_schedulers['scheduler'].state_dict(),
            },
            Path(join(output_dir, "checkpoint.pth")),
        )

        # epoch별 모델 저장
        torch.save({
            "model": model.state_dict(),
            "discriminator": disc.state_dict()
        }, Path(join(output_dir, f"{iepoch}epoch.pth")))

        # 5. Save and log the model and update the link to the best model
        # if iepoch % 5 == 0:
        #     torch.save(model.module.state_dict(), Path(join(output_dir, f"{iepoch}epoch.pth")))

        # Creates a sym link latest.pth -> {iepoch}epoch.pth
        p = Path(join(output_dir, "latest.pth"))
        if p.is_symlink() or p.exists():
            p.unlink()
        p.symlink_to(f"{iepoch}epoch.pth")

        _improved = []
        for _phase, k, _mode in train_args.best_model_criterion:
            # e.g. _phase, k, _mode = "train", "loss", "min"
            if reporter.has(_phase, k):
                best_epoch = reporter.get_best_epoch(_phase, k, _mode)
                # Creates sym links if it's the best result
                if best_epoch == iepoch:
                    p = Path(join(output_dir, f"{_phase}.{k}.best.pth"))
                    if p.is_symlink() or p.exists():
                        p.unlink()
                    p.symlink_to(f"{iepoch}epoch.pth")
                    _improved.append(f"{_phase}.{k}")
        if len(_improved) == 0:
            logging.info("There are no improvements in this epoch")
        else:
            logging.info(
                "The best model has been updated: " + ", ".join(_improved)
            )

        # 6. Remove the model files excluding n-best epoch and latest epoch
        _removed = []
        # Get the union set of the n-best among multiple criterion
        nbests = set().union(
            *[
                set(reporter.sort_epochs(ph, k, m)[: max(keep_nbest_models)])
                for ph, k, m in train_args.best_model_criterion
                if reporter.has(ph, k)
            ]
        )

        # Generated n-best averaged model
        if (
                train_args.nbest_averaging_interval > 0
                and iepoch % train_args.nbest_averaging_interval == 0
        ):
            average_nbest_models(
                reporter=reporter,
                output_dir=output_dir,
                best_model_criterion=train_args.best_model_criterion,
                nbest=keep_nbest_models,
                suffix=f"till{iepoch}epoch",
            )

        for e in range(1, iepoch):
            p = Path(join(output_dir, f"{e}epoch.pth"))
            if p.exists() and e not in nbests:
                p.unlink()
                _removed.append(str(p))
        if len(_removed) != 0:
            logging.info("The model files were removed: " + ", ".join(_removed))

        # 7. If any updating haven't happened, stops the training
        if all_steps_are_invalid:
            logging.warning(
                f"The gradients at all steps are invalid in this epoch. "
                f"Something seems wrong. This training was stopped at {iepoch}epoch"
            )
            break

        # 8. Check early stopping
        if train_args.patience is not None:
            if reporter.check_early_stopping(
                    train_args.patience, *train_args.early_stopping_criterion
            ):
                break
    else:
        logging.info(
            f"The training was finished at {args.max_epoch} epochs "
        )


def _fs_step(GT_mel, pred_mel, fs2_loss, discriminator, adversarial_loss, condition):
    losses = {}
    losses["total_loss"] = fs2_loss
    # if self.config.compute_adversarial_loss:
    generator_adversarial_loss, fm_loss = adversarial_loss.generator_loss(
        GT_mel, pred_mel, condition, discriminator.to('cuda:0'),
    )
    if fm_loss > 0:
        fm_alpha = (losses["total_loss"] / fm_loss).detach()
        fm_loss = fm_alpha * fm_loss
    losses["total_loss"] += generator_adversarial_loss + fm_loss
    losses["adv_g_loss"] = generator_adversarial_loss
    losses["fm_loss"] = fm_loss

    return losses["total_loss"]


def _ds_step(GT_mel, pred_mel, discriminator, adversarial_loss, condition):
    loss = adversarial_loss.discriminator_loss(
        GT_mel, pred_mel, condition, discriminator.to('cuda:0')
    )

    return loss


def train_one_epoch(model, optimizers, schedulers, iterator, reporter, summary_writer, options, distributed, ngpu,
                    epoch, disc, advloss, disc_optimizers, disc_schedulerss):
    grad_noise = options.grad_noise
    accum_grad = options.accum_grad
    grad_clip = options.grad_clip
    grad_clip_type = options.grad_clip_type
    log_interval = options.log_interval
    distributed = distributed

    with open('/home/lee/Desktop/INSUNG/intensity_fs2/tts_tool/tts/fastspeech2/intensity.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model.intensity_model = RankModel(config).to('cuda:0')

    model.intensity_model.load_state_dict(torch.load(
        # '/home/lee/Desktop/INSUNG/intensity_fs2/tts_tool/intensity_train/runs/EGCA/model_step_23000.pth',
        '/home/lee/Desktop/INSUNG/intensity_fs2/tts_tool/intensity_train/runs/context_EGCA/model_step_23000.pth',
        map_location='cuda:0'))
    for p in model.intensity_model.parameters():
        p.requires_grad = False

    # optimizers[0].param_groups[0]['capturable'] = True
    ngpu = ngpu

    if log_interval is None:
        try:
            log_interval = max(len(iterator) // 20, 10)
        except TypeError:
            log_interval = 100

    model.train()
    disc.train()
    all_steps_are_invalid = True
    # [For distributed] Because iteration counts are not always equals between
    # processes, send stop-flag to the other processes if iterator is finished
    iterator_stop = torch.tensor(0).to(device)

    start_time = time.perf_counter()
    for iiter, batch in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
    ):
        assert isinstance(batch, dict), type(batch)

        if distributed:
            torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
            if iterator_stop > 0:
                break

        torch.cuda.empty_cache()
        batch = to_device(batch, device)

        with reporter.measure_time("forward_time"):
            retval = model(**batch)
            retval_disc = model(**batch)

            # Note(kamo):
            # Supporting two patterns for the returned value from the model
            #   a. dict type
            if isinstance(retval, dict):
                loss = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                pred_mel = retval['after_outs']
                intensity = retval["condition"]
                optim_idx = retval.get("optim_idx")

                loss_for_disc = retval_disc["loss"]
                pred_mel_for_disc = retval_disc['after_outs']
                intensity_for_disc = retval_disc["condition"]

                if optim_idx is not None and not isinstance(optim_idx, int):
                    if not isinstance(optim_idx, torch.Tensor):
                        raise RuntimeError(
                            "optim_idx must be int or 1dim torch.Tensor, "
                            f"but got {type(optim_idx)}"
                        )
                    if optim_idx.dim() >= 2:
                        raise RuntimeError(
                            "optim_idx must be int or 1dim torch.Tensor, "
                            f"but got {optim_idx.dim()}dim tensor"
                        )
                    if optim_idx.dim() == 1:
                        for v in optim_idx:
                            if v != optim_idx[0]:
                                raise RuntimeError(
                                    "optim_idx must be 1dim tensor "
                                    "having same values for all entries"
                                )
                        optim_idx = optim_idx[0].item()
                    else:
                        optim_idx = optim_idx.item()

            #   b. tuple or list type
            else:
                loss = retval[0]
                stats = retval[1]
                weight = retval[2]
                pred_mel = retval[3]
                intensity = retval[4]
                # optim_idx = retval.get("optim_idx")

                # loss_for_disc = retval_disc[0]
                pred_mel_for_disc = retval_disc[3]
                intensity_for_disc = retval_disc[4]
                optim_idx = None

        stats = {k: v for k, v in stats.items() if v is not None}

        if ngpu > 1 or distributed:
            # Apply weighted averaging for loss and stats
            loss = (loss * weight.type(loss.dtype)).sum()

            # if distributed, this method can also apply all_reduce()
            stats, weight = recursive_average(stats, weight, distributed)

            # Now weight is summation over all workers
            loss /= weight
        if distributed:
            # NOTE(kamo): Multiply world_size because DistributedDataParallel
            # automatically normalizes the gradient by world_size.
            loss *= torch.distributed.get_world_size()

        loss /= accum_grad

        generator_loss = _fs_step(batch['feats'], pred_mel, loss, discriminator=disc, adversarial_loss=advloss,
                                  condition=intensity)
        discriminator_loss = _ds_step(batch['feats'], pred_mel_for_disc, discriminator=disc, adversarial_loss=advloss,
                                      condition=intensity_for_disc)

        stats['gen_loss'] = generator_loss
        stats['disc_loss'] = discriminator_loss
        reporter.register(stats, weight)

        with reporter.measure_time("backward_time"):
            generator_loss.backward()

            # disc_weights_before = {name: param.clone().detach() for name, param in disc.named_parameters()}
            #
            # # 생성자 역전파
            # generator_loss.backward()
            #
            # # 생성자 역전파 후, 판별자 역전파 전에 판별자 가중치 확인
            # disc_weights_changed = False
            # for name, param in disc.named_parameters():
            #     if not torch.allclose(param, disc_weights_before[name]):
            #         print(f"생성자 학습 후 판별자 가중치 '{name}'가 변경됨")
            #         disc_weights_changed = True
            #
            # if not disc_weights_changed:
            #     print("생성자 학습 후에도 판별자 가중치가 변경되지 않음")

            discriminator_loss.backward()

        if iiter % accum_grad == 0:
            # gradient noise injection
            if grad_noise:
                add_gradient_noise(
                    model,
                    reporter.get_total_count(),
                    duration=100,
                    eta=1.0,
                    scale_factor=0.55,
                )

            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
                norm_type=grad_clip_type,
            )
            # PyTorch<=1.4, clip_grad_norm_ returns float value
            if not isinstance(grad_norm, torch.Tensor):
                grad_norm = torch.tensor(grad_norm)

            if not torch.isfinite(grad_norm):
                logging.warning(
                    f"The grad norm is {grad_norm}. Skipping updating the model."
                )

            else:
                all_steps_are_invalid = False
                with reporter.measure_time("optim_step_time"):
                    for iopt, (optimizer, scheduler) in enumerate(
                            zip(optimizers, schedulers)
                    ):
                        if optim_idx is not None and iopt != optim_idx:
                            continue
                        # optimizer.param_groups[0]['capturable'] = True
                        optimizer.step()
                        disc_optimizers.step()

                        if isinstance(scheduler, AbsBatchStepScheduler):
                            scheduler.step()
                            disc_schedulerss['scheduler'].step()
            for iopt, optimizer in enumerate(optimizers):
                if optim_idx is not None and iopt != optim_idx:
                    continue
                optimizer.zero_grad()
                disc_optimizers.zero_grad()

            # Register lr and train/load time[sec/step],
            # where step refers to accum_grad * mini-batch
            reporter.register(
                dict(
                    {
                        f"optim{i}_lr{j}": pg["lr"]
                        for i, optimizer in enumerate(optimizers)
                        for j, pg in enumerate(optimizer.param_groups)
                        if "lr" in pg
                    },
                    train_time=time.perf_counter() - start_time,
                ),
            )
            reporter.register(
                dict(
                    {
                        f"disc_optim{i}_lr{j}": pg["lr"]
                        for i, disc_optimizer in enumerate([disc_optimizers])
                        for j, pg in enumerate(disc_optimizer.param_groups)
                        if "lr" in pg
                    },
                    disc_train_time=time.perf_counter() - start_time,
                ),
            )
            start_time = time.perf_counter()

        # NOTE(kamo): Call log_message() after next()
        reporter.next()
        if iiter % log_interval == 0:
            logging.info(reporter.log_message(-log_interval))
            if summary_writer is not None:
                reporter.tensorboard_add_scalar(summary_writer, -log_interval)
    else:
        if distributed:
            iterator_stop.fill_(1)
            torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

    return all_steps_are_invalid


def validate_one_epoch(model, iterator, reporter, options, distributed, ngpu, disc, advloss):
    ngpu = ngpu
    no_forward_run = options.no_forward_run

    model.eval()
    disc.eval()  # discriminator도 eval 모드로

    iterator_stop = torch.tensor(0).to("cuda:0" if ngpu > 0 else "cpu")
    for batch in iterator:
        assert isinstance(batch, dict), type(batch)
        if distributed:
            torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
            if iterator_stop > 0:
                break

        batch = to_device(batch, "cuda:0" if ngpu > 0 else "cpu")
        if no_forward_run:
            continue

        # Forward pass for both generator and discriminator
        retval = model(**batch)
        # if isinstance(retval, dict):
        stats = retval[1]
        weight = retval[2]
        pred_mel = retval[3]
        intensity = retval[4]

        # Generator와 Discriminator loss 계산 (validation에서는 backprop 없이)
        with torch.no_grad():
            # Generator loss
            generator_loss = _fs_step(batch['feats'], pred_mel, retval[0],
                                      discriminator=disc,
                                      adversarial_loss=advloss,
                                      condition=intensity)

            # Discriminator loss
            discriminator_loss = _ds_step(batch['feats'], pred_mel,
                                          discriminator=disc,
                                          adversarial_loss=advloss,
                                          condition=intensity)

            # Add adversarial losses to stats
            stats["val_generator_loss"] = generator_loss.item()
            stats["val_discriminator_loss"] = discriminator_loss.item()

        if ngpu > 1 or distributed:
            stats, weight = recursive_average(stats, weight, distributed)

        reporter.register(stats, weight)
        reporter.next()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch TTS Training')
    parser.add_argument('--max_epoch', default=1000, type=int,
                        help='GPU id to use.')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu_num', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument("--preprocess_config", type=str, default="./config/esd/preprocess.yaml",
                        help="path to preprocess.yaml", )
    parser.add_argument("--model_config", type=str, default="./config/esd/model_8L.yaml",
                        help="path to model.yaml")
    parser.add_argument("--option_config", type=str, default="./config/esd/option.yaml",
                        help="path to option.yaml")
    parser.add_argument("--log_path", type=str, default="./model_checkpoint/mamba1_context_EGCA_JCU/log",
                        help="path to log")
    parser.add_argument("--result_path", type=str, default="./model_checkpoint/mamba1_context_EGCA_JCU/",
                        help="path to result checkpoint")
    parser.add_argument("--resume", type=bool, default=False,
                        help="path to result checkpoint")
    args = parser.parse_args()

    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    option_config = yaml.load(open(args.option_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, option_config)

    main(args, configs)
