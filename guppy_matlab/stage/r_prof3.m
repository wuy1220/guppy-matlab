function r_prof3()
%R_PROF3 Function-level attribution for a single-token forward.
%   Only TotalTime is touched: field names differ between releases and a
%   stale m-file cache served the previous version twice.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
out = fullfile(root, 'logs', 'prof_infer.txt');

S = load(fullfile(root, 'checkpoints', 'full3_10000.mat'));
P = S.P; cfg = S.cfg;
ids = ones(1, 1, 'int32');

for k = 1:20, guppy.forward(P, cfg, ids, [], false); end

profile off; profile clear;
profile on;
for k = 1:100, guppy.forward(P, cfg, ids, [], false); end
profile off;

p  = profile('info');
Fn = p.FunctionTable;
tot = [Fn.TotalTime];
[~, ord] = sort(tot, 'descend');

fid = fopen(out, 'w');
fprintf(fid, 'fields: %s\n\n', strjoin(fieldnames(Fn)', ' '));
fprintf(fid, '%-44s %10s %8s\n', 'function', 'total_ms', 'calls');
for i = 1:min(20, numel(ord))
    j = ord(i);
    n = 1;
    if isfield(Fn, 'NumCalls'), n = Fn(j).NumCalls; end
    fprintf(fid, '%-44s %10.2f %8d\n', Fn(j).FunctionName, tot(j) * 1000, n);
end
fclose(fid);

type(out);
fprintf('per-forward total from profiler: %.3f ms\n', ...
    tot(strcmp({Fn.FunctionName}, 'guppy.forward')) * 10);
end
