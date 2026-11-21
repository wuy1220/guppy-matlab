function drift2()
%DRIFT2 Second look: is the slowdown permanent, or does it reset?
%   Run 1 showed 0.71-0.81 s/step for eight blocks, a 1.59 s spike, then a
%   permanent 1.04 plateau. Two candidates: the machine changed under us, or
%   MATLAB ran out of headroom. Printing the memory picture next to the series
%   separates them.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
cfg = guppy.defaultConfig('steps', 100, 'warmup', 10);

m = memory;
fprintf('before load: MATLAB used %.0f MB, max block %.0f MB, system free %.0f MB\n', ...
    m.MemUsedMATLAB/1e6, m.MaxPossibleArrayBytes/1e6, freeSystemMB());

D = data.loadData(cfg);
rng(0);
idx = randperm(size(D.train.ids, 1), cfg.batch);
[X, Y] = data.getBatch(D.train.ids, D.train.len, idx);
fprintf('T=%d  N=%d tokens\n', size(X,1), size(X,1)*cfg.batch);

P = guppy.initParams(cfg, 'single'); st = struct();
for w = 1:3, [P, st] = guppy.step(P, st, X, Y, cfg); end

fprintf('\nblock  s/step  rel   MATLAB_MB  free_MB\n');
t0 = NaN;
for k = 1:10
    t = tic;
    for j = 1:5, [P, st] = guppy.step(P, st, X, Y, cfg); end
    el = toc(t) / 5;
    if isnan(t0), t0 = el; end
    m = memory;
    fprintf('%5d  %6.3f  %4.2fx  %9.0f  %7.0f\n', k, el, el/t0, m.MemUsedMATLAB/1e6, freeSystemMB());
end
end

function mb = freeSystemMB()
mb = NaN;
try
    [~, out] = system(['powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"']);
    mb = str2double(strtrim(out)) / 1024;
catch
end
end
