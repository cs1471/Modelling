
%%% This is a tutorial and exercises for image statistics.
%%% We will look at the distributions of filter
%%% activations to images, and also generate synthetic
%%% multiplicative distributions according to the Gaussian
%%% Scale Mixture model.

%%% Make sure to add paths of all directories associated with
%%% the tutorial. Also, note that some functions used are
%%% similar to built in Matlab functions -- you can use these
%%% or your own versions.

%%% Read the comments and copy each line of code into matlab.
%%% Note that in some places the code is incomplete and you
%%% need to fill in the pieces...
%%% Type 'help <function_name>' in Matlab window for any
%%% function you would like more information on.
%%% Type 'which <function_name>' in Matlab window for the
%%% location of a file.

%%% OS July 2011; as part of Berkeley summer course.

%%% July 2013; just using sections 1 and 3.

%%% Odelia Schwartz, Berkeley course 2016

%%% (1). Images and statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% (a) Read an image and look at marginal statistics.
%%% As in the first lab, we will use the steerable
%%% pyramid to look at oriented filter outputs to
%%% an image. You can try different images and
%%% subbands. Here is an example.

im = pgmRead('crowd.pgm');
figure(1); showIm(im);
numlevs = 3;
numoris = 3; % this is the number of orientations minus one
[pyr,pind] = buildSFpyr(im, numlevs, numoris);

%%% *To do*: Take a subband of the pyramid, such as the
%%% first scale and orientation (call it nbr1).
nbr1 = spyrBand(pyr,pind,1,1);

%%% *To do*: Plot the marginal distribution of the
%%% subband (e.g., the distribution of nbr1(:))
%%% You can use the histo function. How does this
%%% compare to a Gaussian? What is the kurtosis?

binsz = 36;
[N,X]=histo(nbr1,binsz);
plot(X,N./sum(N))
kurtosis(nbr1(:))

%%% (b) Look at joint statistics
%%% *To do*: Examine pairs of filter outputs
%%% (which we will call nbr1 and nbr2) to the same image,
%%% and their joint statistics. Try different filter
%%% pairs (different orientations and a spatially
%%% shifted version of nbr1 using the shift function
%%% with different amounts of shift).

nbr2 = spyrBand(pyr,pind,1,3);

%% Shifted.

nbr2 = shift(nbr1, [5 0]);

%%% Plot joint conditional histograms (bowties)
%%% of the dependencies between nbr1 and nbr2.
%%% You can use the function bowtie.m. Here we include
%%% the complete code. Note that it is a conditional
%%% distribution and so not symmetric.

%mybowtie(nbr1, nbr2)
binsz=51;
[H,Y,X] = jhisto(nbr1, nbr2, binsz);
colmax = max(1,max(H));
H = H ./ (ones(size(H,1),1)*colmax);
imagesc(X,Y,H); axis('xy'); axis('square');
colormap('gray');

%%% Plot contour plots of the dependencies
%%% and compare joint with multiplication of
%%% the marginals (why?). Below is an example for
%%% the joint dependency. You can vary the bin size
%%% and parameters.
[N, Y, X] = jhisto(nbr1, nbr2, 32);
N=N./sum(sum(N));
subplot(2,2,1)
theval=max(max(N));
the_epsilon = 10^-5;
contour(X,Y,log2(N+the_epsilon),8)
axis('square'); colormap('gray');
colormap('gray');

%%% *To do*: compare the joint histogram to the product of marginals.
%%% That is, compute a 1D marginal histogram for nbr1 and for nbr2
%%% (using, e.g., histo). Take the product of the two 1D marginal
%%% histograms. Plot a contour plot of this product of marginals
%%% and compare to the contour plot above (are these independent?)
%%% You can look at different example subbands -- the shapes can
%%% change based on orientation, spatial distance, of nbr1
%%% and nbr2.

[h,x]=histo(nbr1, 32);
[h2,x2]=histo(nbr2, 32);
h=h./sum(h);
h2=h2./sum(h2);
prod_marginals=h'*h2;
subplot(2,2,2)
the_epsilon = .00001;
contour(x2,x,log2(prod_marginals+the_epsilon),8)
axis('square'); colormap('gray');

%% If you have time at the end, you can repeat these
%% stats for filter outputs derived from ICA

%%% (2). Synthetic distributions: multiplicative dependencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Simple version of Gaussian Scale Mixture.
%%% Generate two Gaussian distributions that
%%% are independent as above.
N=100000;
g1 = randn(1,N);
g2 = randn(1,N);

%%% Generate a mixer distribution. We will use a Rayleigh
%%% distribution for the mixer. Matlab has a built in function
%%% to generate samples from this distribution:
thev=raylrnd(1, [1 N]);
%%% Look at the mixer distribution:
figure; clf;
binsz = 32;
[H,X]=histo(thev,binsz);
plot(X, H./sum(H));
%%% The mixer variable thev takes the form v .* exp(-v.^2/2);
%%% where the values of v are always positive.

%%% Generate a GSM by multiplying both Gaussian sources (point by point)
%%% by the common mixer variable:
x1 = thev.* g1;
x2 = thev .* g2;

%%% *To do*: This generates a multiplicative bowtie dependency,
%%% similar to what we saw for the images. Verify by looking at the
%%% bowtie function for x1 and x2 (as with the images above).
%%% Also look at the bowtie between g1 and g2; these should look
%%% independent.

mybowtie(x1,x2);
mybowtie(g1,g2);

%%% *To do*: Plot the marginal distribution of x1.
%%% How does it compare to a Gaussian?
[H,X] = histo(x1,-.2);
hold on;
plot(X,H./sum(H));    
[H,X] = histo(g1,-.2);
plot(X,H./sum(H),'r');
hold off;
kurtosis(g1) 
kurtosis(x1)

%%% *To do*: Try ICA on [x1; x2] using the ica4 function -- are
%%% the dependencies reduced?
clf;
[newsig, mat, evals]=ica4([x1; x2]);
mybowtie(newsig(1,:), newsig(2,:))

%%% Since we have generated a multiplicative dependency, we can
%%% estimate the original Gaussian variables (g1 and g2) given
%%% x1 and x2. More specifically, estimating the Gaussian
%%% amounts to a form of division. This is done in practice through
%%% Bayesian inference (since we have set a prior on the mixer and
%%% Gaussian distribution). We use the function gaussEstim.m to
%%% estimate the Gaussian component g1 corresponding to x1
%%% and g2 corresponding to x2.

numxis = 2;

%%% Compute the square root of the sum of squares of the xi
nGrp1 = 0;
for i=1:numxis
 nGrp1 = nGrp1 + (eval(sprintf('x%d', i))).^2;
 end
nGrp1 = sqrt(nGrp1);
%%% Estimate the Gaussians
estimg1 = gaussEstim(nGrp1, x1, numxis, 1);
estimg2 = gaussEstim(nGrp1, x2, numxis, 1);

%%% If our estimates are good, then we should get back a Gaussian.
%%% *To do* Look at the resulting 1D distribution of estimg1 in comparison
%%% to a Gaussian. Also, if our estimates are good, the estimated
%%% Gaussians should be independent. Look at the bowtie dependency
%%% between estimg1 and estimg2. Why is the fit not good?
[H,X] = histo(estimg1,-.2);
plot(X,H./sum(H));    
kurtosis(estimg1)
mybowtie(estimg1, estimg2);

%%% *To do*: Create 100 xi by multiplting 100
%%% independent Gaussians by the same common mixer.
%%% Estimate the Gaussian g1 and g2 based on all
%%% other 100. Is the fit better? Are the estimated
%%% Gaussians independent?

numxis = 100;
for i=1:numxis
assignin('base', sprintf('randn%d', i), randn(size(g1)));
assignin('base', sprintf('x%d', i), thev.* eval(sprintf('randn%d', i)));
end
nGrp1 = 0;
for i=1:numxis
 nGrp1 = nGrp1 + (eval(sprintf('x%d', i))).^2;
 end
nGrp1 = sqrt(nGrp1);
estimg1 = gaussEstim(nGrp1, x1, numxis, 1);
estimg2 = gaussEstim(nGrp1, x2, numxis, 1);
[H,X] = histo(estimg1,-.2);
plot(X,H./sum(H));    
kurtosis(estimg1)
mybowtie(estimg1,estimg2)

%%% *To do*: Create 50 xi by multiplying 50
%%% independent Gaussians by the same common mixer.
%%% Create another 50 xi by multiplying 50
%%% independent Gaussians by the same common mixer
%%% (but different from the mixer you used for the
%%% first 50). Estimate the Gaussian estimg1 and estimg2
%%% making the (wrong) assumption that all 100
%%% xi were generated with the same common mixer.
%%% Is the fit good? Why or why not?

thev1=raylrnd(1, [1 N]);
thev2=raylrnd(1, [1 N]);
numxis = 50;
for i=1:numxis
assignin('base', sprintf('randn%d', i), randn(size(g1)));
assignin('base', sprintf('x%d', i), thev1.* eval(sprintf('randn%d', i)));
end
for i=51:100
assignin('base', sprintf('randn%d', i), randn(size(g1)));
assignin('base', sprintf('x%d', i), thev2.* eval(sprintf('randn%d', i)));
end
mybowtie(x1,x2); 
mybowtie(x1,x51);

nGrp1 = 0;
for i=1:100
 nGrp1 = nGrp1 + (eval(sprintf('x%d', i))).^2;
 end
nGrp1 = sqrt(nGrp1);
estimg1 = gaussEstim(nGrp1, x1, numxis, 1);
estimg2 = gaussEstim(nGrp1, x2, numxis, 1);
[H,X] = histo(estimg1,-.2);
plot(X,H./sum(H));
kurtosis(estimg1)
mybowtie(estimg1,estimg2)

%% If you have time, you can try this for image subbands.
%% But note: A GSM model with a single mixer (v) is not
%% a good description in general.

