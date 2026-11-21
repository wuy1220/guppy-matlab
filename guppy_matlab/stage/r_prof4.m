function r_prof4()
%R_PROF4 Line-level attribution inside guppy.forward at T=1.
%   The function profile says 84% of the time never leaves forward.m itself,
%   so the answer has to come from ExecutedLines, not the call tree.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
out = fullfile(root, 'logs', 'prof_lines.txt');

S = load(fullfile(root, 'checkpoints', 'full3_10000.mat'));
P = S.P; cfg = S.cfg;
ids = ones(1, 1, 'int32');

for k = 1:20, guppy.forward(P, cfg, ids, [], false); end

profile off; profile clear; profile on;
for k = 1:100, guppy.forward(P, cfg, ids, [], false); end
profile off;

p  = profile('info');
Fn = p.FunctionTable;
i  = find(strcmp({Fn.FunctionName}, 'forward'), 1);
el = Fn(i).ExecutedLines;
L  = splitlines(fileread(Fn(i).FileName));
nCall = Fn(i).NumCalls;

fid = fopen(out, 'w');
fprintf(fid, 'forward.m  %d calls, total %.1f ms\n\n', nCall, Fn(i).TotalTime * 1000);
[~, o] = sort(el(:, 3), 'descend');
for q = 1:min(22, size(el, 1))
    ln = el(o(q), 1);
    tt = el(o(q), 3);
    if ln >= 1 && ln <= numel(L), txt = strtrim(L{ln}); else, txt = '?'; end
    fprintf(fid, 'L%-4d %8.2f ms  (%5.1f%%)  %s\n', ln, tt * 1000, ...
        100 * tt / Fn(i).TotalTime, txt(1:min(78, numel(txt))));
end
fclose(fid);
type(out);
end
