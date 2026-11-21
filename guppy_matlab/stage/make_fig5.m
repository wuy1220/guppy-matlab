function make_fig5()
%MAKE_FIG5 The inference figure: where a token's cost goes, and what a cache
%   does to it.
%
%   fig5a  breakdown of one stateless forward at T=1, from the line profile
%   fig5b  per-token stopwatch over 110 generated tokens, both paths
%
%   Sizes follow the same rule as make_figs: layout at roughly the printed
%   width (~7 inch) so the font sizes on screen are the ones that reach paper.

root   = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
outDir = fullfile(root, 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

f = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 700 300]);

%% ---------------- a: where 3.56 ms goes ----------------
item = {'weight GEMMs', 'cache (unused)', 'attention', 'LayerNorm', 'the rest'};
ms   = [2.01, 0.58, 0.35, 0.32, 0.30];
waste = [false true false false false];

ax = axes('Parent', f, 'Position', [0.185 0.155 0.325 0.70]);
hold(ax, 'on');
for i = 1:numel(ms)
    if waste(i)
        barh(ax, i, ms(i), 0.58, 'FaceColor', [0.855 0.330 0.275], 'EdgeColor', 'none');
    else
        barh(ax, i, ms(i), 0.58, 'FaceColor', [0.400 0.545 0.720], 'EdgeColor', 'none');
    end
end
ax.YDir = 'reverse';
ax.YTick = 1:numel(ms);
ax.YTickLabel = item;
ax.YLim = [0.4 numel(ms) + 0.6];
ax.XLim = [0 2.78];
ax.XGrid = 'on'; ax.GridAlpha = 0.13; ax.Box = 'off';
ax.FontSize = 7.6; ax.TickDir = 'out'; ax.Layer = 'bottom';
xlabel(ax, 'milliseconds', 'FontSize', 7.6);
title(ax, 'a   one token, 3.56 ms', 'FontSize', 8.6, 'FontWeight', 'bold');
for i = 1:numel(ms)
    text(ax, ms(i) + 0.06, i, sprintf('%.2f  (%.0f%%)', ms(i), 100*ms(i)/3.56), ...
        'FontSize', 7.2, 'VerticalAlignment', 'middle', 'Color', [0.13 0.13 0.13]);
end

%% ---------------- b: per-token stopwatch ----------------
D = readmatrix(fullfile(root, 'logs', 'kvc_stepwise.csv'));
step = D(:, 1);  cMs = D(:, 2);  sMs = D(:, 3);
cS = movmean(cMs, 7);
sS = movmean(sMs, 7);

ax2 = axes('Parent', f, 'Position', [0.615 0.155 0.355 0.70]);
hold(ax2, 'on');
h1 = plot(ax2, step, sS, '-', 'Color', [0.855 0.330 0.275], 'LineWidth', 1.5);
h2 = plot(ax2, step, cS, '-', 'Color', [0.180 0.530 0.360], 'LineWidth', 1.6);
yline(ax2, 3.56, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
ax2.XLim = [step(1) step(end)];
ax2.YLim = [0 14];
ax2.XGrid = 'on'; ax2.YGrid = 'on';
ax2.GridAlpha = 0.13; ax2.Box = 'off';
ax2.FontSize = 7.6; ax2.TickDir = 'out'; ax2.Layer = 'bottom';
xlabel(ax2, 'token position', 'FontSize', 7.6);
ylabel(ax2, 'ms per token', 'FontSize', 7.6);
title(ax2, 'b   110 tokens, same weights', 'FontSize', 8.6, 'FontWeight', 'bold');
legend([h1 h2], {'recompute prefix', 'KV cache'}, 'Location', 'northwest', ...
    'FontSize', 7.2, 'Box', 'off');
text(ax2, step(6), 2.72, 'weights read floor', 'FontSize', 6.6, ...
    'Color', [0.42 0.42 0.42]);
text(ax2, step(end) - 3, sS(end) - 1.5, sprintf('%.1f ms', sS(end)), ...
    'FontSize', 7.2, 'Color', [0.855 0.330 0.275], 'HorizontalAlignment', 'right');
text(ax2, step(end) - 3, cS(end) + 0.9, sprintf('%.1f ms', cS(end)), ...
    'FontSize', 7.2, 'Color', [0.180 0.530 0.360], 'HorizontalAlignment', 'right');

exportgraphics(f, fullfile(outDir, 'fig5_inference.png'), 'Resolution', 300);
fprintf('wrote fig5_inference.png\n');
fprintf('cached %.2f -> %.2f ms | stateless %.2f -> %.2f ms | final ratio %.2fx\n', ...
    cS(1), cS(end), sS(1), sS(end), sS(end)/cS(end));
end
