% AGENT_START  File-based command agent for the running MATLAB session.
%
% The MathWorks licence service is unreachable (error 5201), so no second
% MATLAB process can start. This script turns the *already running* session
% into a command executor: any *.m dropped into agent/inbox is evaluated in
% the base workspace and its console output (and error, if any) is written to
% agent/outbox/<same name>.txt.
%
% Stop with:  create agent/STOP  (or press Ctrl+C)

root   = fileparts(mfilename('fullpath'));
inbox  = fullfile(root, 'agent', 'inbox');
outbox = fullfile(root, 'agent', 'outbox');
if ~exist(inbox,  'dir'), mkdir(inbox);  end
if ~exist(outbox, 'dir'), mkdir(outbox); end
stopFlag = fullfile(root, 'agent', 'STOP');
if exist(stopFlag, 'file'), delete(stopFlag); end

hb = fullfile(root, 'agent', 'heartbeat.txt');
fid = fopen(hb, 'w'); fprintf(fid, 'START %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS')); fclose(fid);

fprintf('\n===== AGENT READY =====\n');
fprintf('inbox : %s\n', inbox);
fprintf('outbox: %s\n', outbox);
fprintf('PID   : %d\n', feature('getpid'));
fprintf('Waiting for commands...\n\n');

nDone = 0;
while ~exist(stopFlag, 'file')
    d = dir(fullfile(inbox, '*.m'));
    if isempty(d)
        % pause() services the event queue, which aborts with "the user
        % interface is busy" when a keystroke is being injected at the same
        % moment. A Java sleep does the same job without touching the UI.
        try
            pause(0.4);
        catch
            java.lang.Thread.sleep(400);
        end
        continue
    end
    [~, ord] = sort({d.name});
    d = d(ord);
    src = fullfile(inbox, d(1).name);
    tag = regexprep(d(1).name, '\.m$', '');
    txt = fileread(src);

    t0 = tic;
    ok = true;
    msg = '';
    try
        out = evalc(txt);
    catch err
        ok = false;
        out = '';
        msg = sprintf('%s\n  at %s:%d', err.message, err.stack(1).name, err.stack(1).line);
    end
    el = toc(t0);

    fid = fopen(fullfile(outbox, [tag '.txt']), 'w');
    fprintf(fid, '=== %s ===\nstatus: %s\nelapsed: %.2f s\n---- output ----\n', ...
        tag, string(ok), el);
    fprintf(fid, '%s\n', out);
    if ~ok
        fprintf(fid, '---- error ----\n%s\n', msg);
    end
    fclose(fid);

    delete(src);
    nDone = nDone + 1;
    fid = fopen(hb, 'w');
    fprintf(fid, 'DONE %d @ %s\n', nDone, datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fclose(fid);
    fprintf('>>> %s  (%.1fs, %s)\n', tag, el, string(ok));
end

fprintf('\n===== AGENT STOPPED after %d commands =====\n', nDone);
