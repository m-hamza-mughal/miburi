import torch


def create_scheduler(args, optimizer, **kwargs):
    num_epochs = args.epochs

    if getattr(args, 'lr_noise', None) is not None:
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    lr_scheduler = None
    if args.lr_policy == "onecyclelr":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr_base,
            total_steps=kwargs["total_steps"],
            pct_start=0.05,
            # div_factor=args.DIV_FACTOR_ONECOS,
            # final_div_factor=args.FIN_DACTOR_ONCCOS,
        )
    elif args.lr_policy == "cycliclr":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr_base / 10,
            max_lr=args.lr_base,
            step_size_up=args.lr_cyclestepsizeup,
            mode="triangular2",
            cycle_momentum=False,
        )
    elif args.lr_policy == "cosinerestart":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = 1,
            T_mult=2,
            eta_min = 1e-8,
            last_epoch=-1,
        )
    elif args.lr_policy == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.decay_epochs,
            gamma=args.decay_rate,
        )
    else:
        raise ValueError(f"Unknown LR policy: {args.lr}")
    
    return lr_scheduler