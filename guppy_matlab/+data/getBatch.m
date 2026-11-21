function [X, Y, T] = getBatch(ids, len, idx)
%GETBATCH Assemble one [T x B] batch with right padding.
%   A sample of length L yields L-1 prediction pairs (the reference shifts by
%   one when building the dataset), so T = max(len)-1 and the target matrix
%   is the same window shifted one step right.

sub = ids(idx, :);
L   = double(len(idx));
T   = max(L) - 1;
X   = sub(:, 1:T).';
Y   = sub(:, 2:T+1).';
end
