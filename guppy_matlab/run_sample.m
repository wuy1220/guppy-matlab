function run_sample(ckpt, outName, temp, topk, maxNew)
%RUN_SAMPLE Generate replies for every prompt in data/prompts.txt.
%   Ids are written to data/gen_<outName>.txt in the same order; decode with
%   `python tools/prompts.py decode data/gen_<outName>.txt`.

if nargin < 1 || isempty(ckpt),    ckpt   = 'full_final';  end
if nargin < 2 || isempty(outName), outName = 'full';       end
if nargin < 3 || isempty(temp),    temp   = 0.7;           end
if nargin < 4 || isempty(topk),    topk   = 50;            end
if nargin < 5 || isempty(maxNew),  maxNew = 40;            end

root = fileparts(mfilename('fullpath'));
if contains(ckpt, filesep) || (numel(ckpt) > 1 && ckpt(2) == ':')
    S = load([ckpt '.mat']);
else
    S = load(fullfile(root, 'checkpoints', [ckpt '.mat']));
end
P = S.P; cfg = S.cfg;

lines = readlines(fullfile(cfg.dataDir, 'prompts.txt'));
fid = fopen(fullfile(cfg.dataDir, ['gen_' outName '.txt']), 'w');
for i = 1:numel(lines)
    s = strtrim(lines(i));
    if s == "", continue; end
    ids = str2double(split(s, ','))';
    out = guppy.generate(P, cfg, ids, maxNew, temp, topk);
    fprintf(fid, '%s\n', strjoin(string(out(:)'), ','));
    fprintf('  prompt %d: %d -> %d tokens\n', i, numel(ids), numel(out));
end
fclose(fid);
fprintf('wrote data/gen_%s.txt\n', outName);
end
