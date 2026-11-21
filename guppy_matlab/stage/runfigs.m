function runfigs()
%RUNFIGS Run make_figs and capture its output to logs/figs.txt.
%   The GUI session has no readable console here, so every diagnostic has to
%   go through a file.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
logp = fullfile(root, 'logs', 'figs.txt');
try
    out = evalc('make_figs');
    fid = fopen(logp, 'w'); fprintf(fid, 'OK\n%s\n', out); fclose(fid);
    fprintf('figs OK\n');
catch err
    fid = fopen(logp, 'w');
    fprintf(fid, 'ERR %s\n', err.message);
    for k = 1:numel(err.stack)
        fprintf(fid, '  %s:%d\n', err.stack(k).name, err.stack(k).line);
    end
    fclose(fid);
    fprintf('figs FAILED: %s\n', err.message);
end
end
