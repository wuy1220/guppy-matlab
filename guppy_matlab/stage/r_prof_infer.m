function r_prof_infer()
%R_PROF_INFER Function-level attribution for a single-token forward.
%   The micro-benchmark budget accounted for only 0.74 ms of a measured
%   3.56 ms, so the remainder is somewhere the parts list does not see.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
S = load(fullfile(root, 'checkpoints', 'full3_10000.mat'));
P = S.P; cfg = S.cfg;
ids = ones(1, 1, 'int32');

for k = 1:20, guppy.forward(P, cfg, ids, [], false); end   % warm

profile off; profile clear;
profile on;
for k = 1:100, guppy.forward(P, cfg, ids, [], false); end
profile off;
p = profile('info');
Fn = p.FunctionTable;

% field names differ between releases: probe once instead of guessing
known = {'TotalTime', 'NumCalls', 'SelfTime', 'TotalRecursiveTime'};
have = known(isfield(Fn(1), known));
fprintf('profile fields present: %s\n', strjoin(fieldnames(Fn(1))', ' '));

tot = [Fn.TotalTime];
[~, ord] = sort(tot, 'descend');
if isfield(Fn(1), 'NumCalls'), nc = [Fn.NumCalls]; else, nc = ones(size(tot)); end
if isfield(Fn(1), 'SelfTime'), slf = [Fn.SelfTime]; else, slf = nan(size(tot)); end

fid = fopen(fullfile(root, 'logs', 'prof_infer.txt'), 'w');
hdr = sprintf('%-46s %10s %8s %10s %10s\n', 'function', 'total_ms', 'calls', 'self_ms', 'per_call_us');
fprintf(fid, '%s', hdr); fprintf('%s', hdr);
for i = 1:min(18, numel(ord))
    j = ord(i);
    per = 1e6 * tot(j) / max(nc(j), 1);
    ln = sprintf('%-46s %10.1f %8d %10.1f %10.1f', ...
        Fn(j).FunctionName, tot(j) * 1000, nc(j), slf(j) * 1000, per);
    fprintf(fid, '%s\n', ln); fprintf('%s\n', ln);
end
fclose(fid);

fprintf('\n--- total accounted ---\n');
fprintf('sum of top-level total times   : %.1f ms\n', tot(strcmp({Fn.FunctionName}, 'guppy.forward')) * 1000 / 100);
