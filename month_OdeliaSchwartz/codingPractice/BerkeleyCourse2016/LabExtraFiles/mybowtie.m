
% function bowtie(n1, n2)
% Plot a bowtie conditional dependency between n1 and n2

function bowtie(n1, n2)

binsz=51;
[H,Y,X] = jhisto(n1, n2, binsz);
colmax = max(1,max(H));
H = H ./ (ones(size(H,1),1)*colmax);
imagesc(X,Y,H); axis('xy'); axis('square');
colormap('gray');

