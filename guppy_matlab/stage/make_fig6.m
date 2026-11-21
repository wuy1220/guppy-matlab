function make_fig6()
%MAKE_FIG6 Batched decoding: the roofline prediction, measured.
%
%   fig6a  tokens/s vs batch size, against perfect linear scaling
%   fig6b  ms per step vs batch size, against the 3.6 ms weight-read floor
%
%   The vertical line sits at B = 30, the point where the arithmetic intensity
%   0.5*B reaches the 15 FLOP/byte balance (509 GFLOPS over a measured
%   34 GB/s triad). If the model is right, the curve bends there and
%   not somewhere else.

root   = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(root, 'figures');
D = readmatrix(fullfile(root, 'data', 'batch_decode.csv'));
B = D(:, 1);  ms = D(:, 2);  tps = D(:, 3);  ai = D(:, 4);

f = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 700 285]);

%% ---------------- a: throughput ----------------
ax = axes('Parent', f, 'Position', [0.085 0.175 0.385 0.68]);
hold(ax, 'on');
hIdeal = plot(ax, B, tps(1) * B, ':', 'Color', [0.55 0.55 0.55], 'LineWidth', 1.2);
hReal  = plot(ax, B, tps, '-o', 'Color', [0.400 0.545 0.720], 'LineWidth', 1.6, ...
              'MarkerFaceColor', 'w', 'MarkerSize', 4.2);
xl = xline(ax, 30, '--', 'Color', [0.855 0.330 0.275], 'LineWidth', 1.0);
ax.XScale = 'log'; ax.YScale = 'log';
ax.XLim = [0.85 75]; ax.YLim = [150 30000];
ax.XTick = [1 2 4 8 16 32 64];
ax.XTickLabel = {'1','2','4','8','16','32','64'};
ax.XGrid = 'on'; ax.YGrid = 'on'; ax.GridAlpha = 0.13; ax.Box = 'off';
ax.FontSize = 7.6; ax.TickDir = 'out'; ax.Layer = 'bottom';
xlabel(ax, 'sequences decoded together (log)', 'FontSize', 7.6);
ylabel(ax, 'tokens per second (log)', 'FontSize', 7.6);
title(ax, 'a   the same 32 MB, shared by B queries', 'FontSize', 8.6, 'FontWeight', 'bold');
legend([hReal hIdeal], {'measured', 'perfect scaling'}, 'Location', 'northwest', ...
    'FontSize', 7.0, 'Box', 'off');
text(ax, 32.5, 470, 'B = 30', 'FontSize', 6.8, 'Color', [0.855 0.330 0.275]);
text(ax, 1.1, tps(1)*1.55, sprintf('%.0f', tps(1)), 'FontSize', 6.8, ...
    'Color', [0.400 0.545 0.720]);

%% ---------------- b: cost per step ----------------
ax2 = axes('Parent', f, 'Position', [0.575 0.175 0.385 0.68]);
hold(ax2, 'on');
plot(ax2, B, ms, '-o', 'Color', [0.180 0.530 0.360], 'LineWidth', 1.6, ...
     'MarkerFaceColor', 'w', 'MarkerSize', 4.2);
yline(ax2, 3.56, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
ax2.XScale = 'log'; ax2.YScale = 'log';
ax2.XLim = [0.85 75]; ax2.YLim = [2 40];
ax2.XTick = [1 2 4 8 16 32 64];
ax2.XTickLabel = {'1','2','4','8','16','32','64'};
ax2.XGrid = 'on'; ax2.YGrid = 'on'; ax2.GridAlpha = 0.13; ax2.Box = 'off';
ax2.FontSize = 7.6; ax2.TickDir = 'out'; ax2.Layer = 'bottom';
xlabel(ax2, 'sequences decoded together (log)', 'FontSize', 7.6);
ylabel(ax2, 'ms per step (log)', 'FontSize', 7.6);
title(ax2, 'b   bandwidth first, then GEMM', 'FontSize', 8.6, 'FontWeight', 'bold');
text(ax2, 1.15, 3.05, '3.56 ms bandwidth floor', 'FontSize', 6.8, 'Color', [0.42 0.42 0.42]);
for i = [1 numel(B)]
    text(ax2, B(i)*1.15, ms(i)*1.12, sprintf('%.1f', ms(i)), 'FontSize', 6.8, ...
        'Color', [0.180 0.530 0.360]);
end

exportgraphics(f, fullfile(outDir, 'fig6_batch.png'), 'Resolution', 300);
fprintf('wrote fig6_batch.png\n');
fprintf('B=1 %.0f tok/s %.2f ms | B=%d %.0f tok/s %.2f ms | gain %.1fx\n', ...
    tps(1), ms(1), B(end), tps(end), ms(end), tps(end)/tps(1));
end
