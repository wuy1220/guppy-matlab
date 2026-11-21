function D = loadData(cfg)
%LOADDATA Read the packed uint16 token matrices produced by tools/prep_data.py.
%   train_ids.u16 is (N, 128) row-major little-endian; fread with [128 N]
%   then a transpose gives the (N, 128) matrix without any per-row parsing.

stride = 128;
D = struct();
for nm = ["train", "eval"]
    p = fullfile(cfg.dataDir, [char(nm) '_ids.u16']);
    N = dir(p).bytes / (stride * 2);
    fid = fopen(p, 'r');
    A = fread(fid, [stride N], 'uint16=>uint16').';
    fclose(fid);
    pl = fullfile(cfg.dataDir, [char(nm) '_len.u16']);
    fid = fopen(pl, 'r');
    L = fread(fid, [1 N], 'uint16=>uint16').';
    fclose(fid);
    D.(char(nm)) = struct('ids', A, 'len', L);
end
D.stride = stride;
end
