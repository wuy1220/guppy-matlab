function make_figs()
%MAKE_FIGS Build the four report figures from measured data, not typed numbers.
%
%   fig1  cover figure: the optimisation ladder plus a real generated reply
%   fig2  training curve: full range on top, the long tail zoomed below
%   fig3  measured GEMM ceiling per shape
%   fig4  waterfall of the ladder
%
%   Figure size matters more than resolution here. The report shows each
%   figure about 16 cm wide on A4, so a figure laid out at 12 inch wide would
%   have its 9 pt labels shrunk to under 5 pt on paper. Every 'Position' below
%   is therefore ~6.5-7.5 inch wide, i.e. roughly the printed width, and the
%   font sizes are the sizes that actually reach the page.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
outDir = fullfile(root, 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

S = load(fullfile(root, 'data', 'bench6.mat'));
mn = min(S.M, [], 2);

%% ---------------- fig1: ladder + dialogue ----------------
lab = {'A0  double, pad 128', 'A1  single, pad 128', 'A2  dynamic T', ...
       'A3  fused kernels', 'A4  no dropout', 'A5  lean AdamW'};

f = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 680 300]);
ax = axes('Parent', f, 'Position', [0.19 0.175 0.325 0.685]);
barh(ax, mn, 0.62, 'FaceColor', [0.345 0.545 0.780], 'EdgeColor', 'none');
ax.YTick = 1:numel(mn);
ax.YTickLabel = lab;
ax.YLim = [0.4 numel(mn) + 0.6];
ax.XLim = [0 max(mn) * 1.40];
ax.XGrid = 'on'; ax.GridAlpha = 0.13; ax.Box = 'off';
ax.FontSize = 7.8; ax.TickDir = 'out'; ax.Layer = 'bottom';
xlabel(ax, 'seconds per step  (batch 32, T = 44)', 'FontSize', 7.8);
title(ax, 'a   one step, six implementations', 'FontSize', 8.8, 'FontWeight', 'bold');
for i = 1:numel(mn)
    if i == 1
        txt = sprintf('%.3f s', mn(i));
    else
        txt = sprintf('%.3f s (%.1f x)', mn(i), mn(1) / mn(i));
    end
    text(ax, mn(i) + max(mn) * 0.03, i, txt, 'FontSize', 7.4, ...
        'VerticalAlignment', 'middle', 'Color', [0.13 0.13 0.13]);
end

ax2 = axes('Parent', f, 'Position', [0.565 0.085 0.42 0.845], 'Visible', 'off');
ax2.XLim = [0 1]; ax2.YLim = [0 1];
rectangle(ax2, 'Position', [0.005 0.02 0.99 0.96], 'Curvature', 0.035, ...
    'FaceColor', [0.10 0.11 0.13], 'EdgeColor', [0.26 0.28 0.31], 'LineWidth', 1.0);
lines = wrapSample(fullfile(root, 'data', 'gen_final_text.txt'), 40);
yl = 0.94;
for i = 1:numel(lines)
    if yl < 0.145, break; end
    if yl < 0.30 && startsWith(lines{i}, 'You>'), break; end
    s = lines{i};
    if isempty(strtrim(s)), continue; end
    if startsWith(s, 'You>')
        col = [0.55 0.78 1.00];
    elseif startsWith(s, 'Guppy>')
        col = [0.63 0.90 0.67];
    else
        col = [0.80 0.82 0.85];
    end
    text(ax2, 0.035, yl, s, 'FontName', 'Consolas', 'FontSize', 7.4, ...
        'Color', col, 'Interpreter', 'none', 'VerticalAlignment', 'top');
    yl = yl - 0.0475;
end
text(ax2, 0.035, 0.055, 'b   held-out prompts, 10 000 steps, T 0.7 / top-k 50', ...
    'FontSize', 6.9, 'Color', [0.70 0.72 0.76], 'Interpreter', 'none');

saveFig(f, fullfile(outDir, 'fig1_cover.png'));

%% ---------------- fig2: training curve ----------------
tr = readTrainLog({fullfile(root,'logs','full.csv'), fullfile(root,'logs','full3.csv')});
ev = readEvalLog(fullfile(root, 'agent', 'outbox', '0032_full3.txt'));

if isempty(ev.step) || isempty(tr.step)
    warning('make_figs:noLog', 'training or eval log missing, fig2 skipped');
else
    f2 = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 760 500]);

    ax = axes('Parent', f2, 'Position', [0.115 0.60 0.865 0.315]);
    semilogy(ax, tr.step, tr.loss, '.', 'Color', [0.66 0.75 0.86], 'MarkerSize', 4);
    hold(ax, 'on');
    semilogy(ax, ev.step, ev.loss, 'o-', 'Color', [0.85 0.30 0.20], 'LineWidth', 1.4, ...
        'MarkerFaceColor', [1 1 1], 'MarkerSize', 4);
    ax.XGrid = 'on'; ax.YGrid = 'on'; ax.GridAlpha = 0.10; ax.Box = 'off';
    ax.FontSize = 7.8; ax.XLim = [0 10000];
    ylabel(ax, 'loss (nats)', 'FontSize', 7.8);
    title(ax, 'a   full trajectory', 'FontSize', 8.6, 'FontWeight', 'bold');
    text(ax, 300, 3.2, sprintf('final held-out %.3f  (perplexity %.2f)', ...
        ev.loss(end), exp(ev.loss(end))), 'FontSize', 7.4, 'Color', [0.55 0.20 0.14]);

    ax = axes('Parent', f2, 'Position', [0.115 0.105 0.865 0.335]);
    lo = 2000;
    kt = tr.step >= lo; ke = ev.step >= lo;
    plot(ax, tr.step(kt), tr.loss(kt), '.', 'Color', [0.66 0.75 0.86], 'MarkerSize', 4);
    hold(ax, 'on');
    plot(ax, ev.step(ke), ev.loss(ke), 'o-', 'Color', [0.85 0.30 0.20], 'LineWidth', 1.4, ...
        'MarkerFaceColor', [1 1 1], 'MarkerSize', 4);
    ax.XGrid = 'on'; ax.YGrid = 'on'; ax.GridAlpha = 0.10; ax.Box = 'off';
    ax.FontSize = 7.8; ax.XLim = [lo 10000]; ax.YLim = [0.30 0.55];
    xlabel(ax, 'optimiser step', 'FontSize', 7.8);
    ylabel(ax, 'loss (nats)', 'FontSize', 7.8);
    title(ax, 'b   the long tail, same data', 'FontSize', 8.6, 'FontWeight', 'bold');
    legend(ax, {'train, 20-step sampling', 'held-out, 3 000 samples'}, ...
        'Location', 'northeast', 'FontSize', 7.2, 'Box', 'off');

    saveFig(f2, fullfile(outDir, 'fig2_training.png'));
end

if ~isfield(S, 'rate')
    warning('make_figs:noRate', 'bench6.mat has no roofline, figs 3-4 skipped');
    return
end

%% ---------------- fig3: roofline ----------------
f3 = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 680 280]);
ax = axes('Parent', f3, 'Position', [0.105 0.245 0.875 0.62]);
r = S.rate(:);
short = {'cube 1024^3', 'QKV proj', 'attn out', 'FFN up', 'FFN down', 'LM head', 'scores', 'A-V'};
bar(ax, [S.cube; r], 0.60, 'FaceColor', [0.42 0.60 0.78], 'EdgeColor', 'none');
hold(ax, 'on');
plot(ax, [0.4 8.6], [S.cube S.cube], '--', 'Color', [0.80 0.25 0.20], 'LineWidth', 1.0);
ax.XTick = 1:8; ax.XTickLabel = short; ax.XTickLabelRotation = 32;
ax.YLim = [0 S.cube * 1.18]; ax.Box = 'off'; ax.YGrid = 'on'; ax.GridAlpha = 0.10;
ax.FontSize = 7.6; ax.TickDir = 'out';
ylabel(ax, 'sgemm throughput (GFLOPS)', 'FontSize', 7.8);
text(ax, 8.5, S.cube * 1.07, sprintf('square-matrix ceiling %.0f', S.cube), ...
    'Color', [0.80 0.25 0.20], 'FontSize', 7.2, 'HorizontalAlignment', 'right');
saveFig(f3, fullfile(outDir, 'fig4_roofline.png'));

%% ---------------- fig4: waterfall ----------------
f4 = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 680 280]);
ax = axes('Parent', f4, 'Position', [0.115 0.275 0.865 0.60]);
vals = mn(1:5);
cats = {'naive', 'single', 'dynamic T', 'fused', 'no dropout', 'final'};
hold(ax, 'on');
bar(ax, 1, vals(1), 0.56, 'BaseValue', 0, 'FaceColor', [0.72 0.74 0.77], 'EdgeColor', 'none');
for i = 2:5
    drop = vals(i-1) - vals(i);
    bar(ax, i, max(drop, 0), 0.56, 'BaseValue', vals(i), ...
        'FaceColor', [0.29 0.65 0.44], 'EdgeColor', 'none');
    text(ax, i, vals(i-1) + max(vals) * 0.04, sprintf('-%.2f', drop), ...
        'HorizontalAlignment', 'center', 'FontSize', 7.2);
end
bar(ax, 6, vals(5), 0.56, 'BaseValue', 0, 'FaceColor', [0.85 0.35 0.25], 'EdgeColor', 'none');
text(ax, 6, vals(5) + max(vals) * 0.04, sprintf('%.2f s', vals(5)), ...
    'HorizontalAlignment', 'center', 'FontSize', 7.2, 'FontWeight', 'bold');
ax.XTick = 1:6; ax.XTickLabel = cats; ax.XTickLabelRotation = 22;
ax.XLim = [0.45 6.55]; ax.YLim = [0 max(vals) * 1.24];
ax.Box = 'off'; ax.YGrid = 'on'; ax.GridAlpha = 0.10; ax.FontSize = 7.6;
ax.TickDir = 'out';
ylabel(ax, 'seconds per step', 'FontSize', 7.8);
saveFig(f4, fullfile(outDir, 'fig3_waterfall.png'));

fprintf('figures written to %s\n', outDir);
end

%% ---------------- helpers ----------------
function saveFig(f, path)
try
    exportgraphics(f, path, 'Resolution', 300);
catch
    print(f, '-dpng', '-r300', path);
end
close(f);
end

function out = wrapSample(path, width)
%WRAPSAMPLE Read the decoded dump and hard-wrap it to the panel width.
if ~exist(path, 'file')
    out = {'You> tell me a joke'; 'Guppy> (sample file missing)'};
    return
end
raw = strsplit(fileread(path), newline);
out = {};
for i = 1:numel(raw)
    ln = deblank(raw{i});
    if isempty(ln), continue; end
    if numel(ln) <= width
        out{end+1} = ln; %#ok<AGROW>
        continue
    end
    lead = '';
    if startsWith(ln, 'Guppy> '), lead = 'Guppy> '; elseif startsWith(ln, 'You> '), lead = ln(1:5); end
    body = strtrim(ln(numel(lead)+1:end));
    while numel(body) > width
        cut = find(body(1:width+1) == ' ', 1, 'last');
        if isempty(cut), cut = width; end
        out{end+1} = [lead body(1:cut-1)]; %#ok<AGROW>
        body = strtrim(body(cut+1:end));
        lead = '       ';
    end
    if ~isempty(body), out{end+1} = [lead body]; end %#ok<AGROW>
end
end

function T = readTrainLog(paths)
step = []; loss = [];
for i = 1:numel(paths)
    if ~exist(paths{i}, 'file'), continue; end
    M = readmatrix(paths{i});
    if isempty(M), continue; end
    step = [step; M(:,1)]; loss = [loss; M(:,2)]; %#ok<AGROW>
end
[step, o] = sort(step); loss = loss(o);
keep = loss > 0;
T = struct('step', step(keep), 'loss', loss(keep));
end

function T = readEvalLog(path)
step = []; loss = [];
if ~exist(path, 'file'), T = struct('step', step, 'loss', loss); return; end
tok = regexp(fileread(path), 'eval @(\d+)\s*:\s*([0-9.]+)', 'tokens');
for i = 1:numel(tok)
    step(end+1, 1) = str2double(tok{i}{1}); %#ok<AGROW>
    loss(end+1, 1) = str2double(tok{i}{2}); %#ok<AGROW>
end
T = struct('step', step, 'loss', loss);
end
