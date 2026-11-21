function W = packData(D, T)
%PACKDATA Concatenate the corpus into one token stream and cut fixed windows.
%
%   Every training batch is padded to T = max(len)-1 of its own members, and a
%   random batch of 32 always contains a long sample, so T sits at ~40 while the
%   average member has 27.4 tokens: 31% of every position is padding that the
%   loss mask discards but the forward and backward still compute.
%
%   Slicing a 1.48 M-token stream into windows of T+1 makes every position a
%   real target, keeps the length mix random (so the gradient is not stratified
%   by length), and -- because <|im_end|> and <|im_start|> stay in the stream --
%   teaches the document boundary as an ordinary token.

len = double(D.train.len); ids = D.train.ids;
N = sum(len);
stream = zeros(N, 1, 'uint16');
p = 0;
for i = 1:numel(len)
    stream(p+1:p+len(i)) = ids(i, 1:len(i));
    p = p + len(i);
end

nW = floor((N - 1) / T);
base = (0:nW-1) * T;
W.x = stream(base + (1:T)');
W.y = stream(base + (2:T+1)');
W.n = nW;
W.T = T;
W.stream = stream;
W.ntok = N;
end
