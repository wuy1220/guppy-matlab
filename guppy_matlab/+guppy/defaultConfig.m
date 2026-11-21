function cfg = defaultConfig(varargin)
%DEFAULTCONFIG Single source of truth for every hyper-parameter.
%   cfg = guppy.defaultConfig()            -> full guppylm configuration
%   cfg = guppy.defaultConfig('L',4,'d',256,'steps',800) -> overrides

cfg = struct();
% --- architecture (guppylm: 6 / 384 / 6 / 768 / learned pos / tied head) ---
cfg.L       = 6;        % transformer blocks
cfg.d       = 384;      % model width
cfg.H       = 6;        % heads
cfg.dff     = 768;      % FFN hidden
cfg.V       = 2418;     % BPE vocab actually present in the released tokenizer
cfg.Tmax    = 128;      % context limit
% --- training -------------------------------------------------------------
cfg.batch   = 32;
cfg.lr      = 3e-4;
cfg.minlr   = 3e-5;
cfg.wd      = 0.1;
cfg.b1      = 0.9;
cfg.b2      = 0.95;
cfg.epsAdam = 1e-8;
cfg.warmup  = 200;
cfg.steps   = 10000;
cfg.clip    = 1.0;
cfg.dropout = 0.1;
cfg.seed    = 42;
cfg.eosId   = 2;         % <|im_end|>
cfg.evalEvery = 200;
cfg.saveEvery = 500;
% --- derived / runtime ----------------------------------------------------
cfg.hd      = cfg.d / cfg.H;
cfg.dtype   = 'single';
cfg.maskNeg = -1e9;     % NOT -Inf: single -Inf - (-Inf) = NaN
cfg.lnEps   = 1e-5;     % matches torch.nn.LayerNorm default
cfg.dataDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data');
% --- optimisation switches, one per row of the ablation table ---------------
%   every one of these is a drop-in replacement that was measured in isolation
%   (tools/micro2.m) before being adopted here
cfg.f = struct( ...
    'normfro',   true, ...   % gradient norm via norm(g,'fro') instead of sum(g.^2,'all')
    'cefuse',    true, ...   % softmax-CE without the full-size log-prob temporary
    'adamfuse',  true, ...   % merge the two parameter updates into one pass
    'randbulk',  true, ...   % rand(n,d,'single') instead of rand(size(X),'like',X)
    'maskcache', true);      % build the causal mask once per sequence length


for k = 1:2:numel(varargin)
    cfg.(varargin{k}) = varargin{k+1};
end
if cfg.d / cfg.H ~= round(cfg.d / cfg.H)
    error('guppy:badShape', 'd must be divisible by H');
end
cfg.hd = cfg.d / cfg.H;
end
