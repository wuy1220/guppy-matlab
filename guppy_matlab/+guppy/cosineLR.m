function lr = cosineLR(t, cfg)
%COSINELR Warmup then cosine decay, identical to guppylm's get_lr().
if t < cfg.warmup
    lr = cfg.lr * t / cfg.warmup;
else
    prog  = (t - cfg.warmup) / max(1, cfg.steps - cfg.warmup);
    coeff = 0.5 * (1 + cos(pi * prog));
    lr    = cfg.minlr + (cfg.lr - cfg.minlr) * coeff;
end
end
