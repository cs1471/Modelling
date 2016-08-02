
%%% path(path,'/Users/schwartzlab/Desktop/BerkeleyCourse2011Stuff/ExtraFiles2012/');

%%% This is a tutorial and exercises for linear visual filters.
%%% It includes both fixed filters using a steerable oriented
%%% pyramid (part 1) and learning filters from an image ensemble
%%% (part 2) using linear transforms such as PCA and ICA.

%%% Read the comments and copy each line of code into matlab.
%%% Note that in some places the code is incomplete and you
%%% need to fill in the pieces...
%%% Type 'help <function_name>' in Matlab window for any
%%% function you would like more information on.
%%% Type 'which <function_name>' in Matlab window for the
%%% location of a file.

%%% Berkeley course 2016, Odelia Schwartz

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% (1) Fixed visual filters. 

%%% Load an image (try different images)
im = pgmRead('einstein.pgm');
figure(1); showIm(im);

%%% (1a) Gabor filters
%%% Make a Gabor gilter by multiplying a sinusoid
%%% with a Gaussian.

% random location
% or center location
fsz = max(2,round(rand*20));
ctr = [round(fsz/2) round(fsz/2)];
theperiod = 8;
thesig=max(2,ceil(round(rand*8)));
direction = sign(.5-rand).*rand*2*pi;
phase = rand*2*pi;
theSine = mkSine(fsz, theperiod, direction, 1, phase, ...
(fsz+1)*[rand rand]);
showIm(theSine)
theGauss = mkGaussian(fsz, thesig, ctr,1);
showIm(theGauss)
theFilt = theSine .* theGauss;
showIm(theFilt)

%%% *To do*:
%%% 1. Try making different Gabor filters by varying parameters
%%% (e.g., size, thesig, direction, phase above.)
%%% 2. Convolve the image with Gabor filters using the conv2
%%% function and 'same' for handling edges.

imnew = conv2(im,theFilt,'same'); showIm(imnew);

%%% (1b) Steerable pyramid filters
% Based on Eero Simoncelli's steerable
% filter code and longer tutorial.
% E P Simoncelli and W T Freeman, 1995: 
% http://www.cns.nyu.edu/~eero/steerpyr/

%% First, we'll see what it means to take a low-pass and high-pass
%% filtering of an image. Then we'll proceed to describing the 
%% steerable pyramid.

binom5 = binomialFilter(5);
lo_filt = binom5*binom5';
blurred = rconv2(im,lo_filt);
subplot(1,2,1); showIm(im, 'auto2', 'auto', 'im');
subplot(1,2,2); showIm(blurred, 'auto2', 'auto', 'blurred');

%% Subtracting the blurred image from the original leaves ONLY the
%% fine scale detail:
fine0 = im - blurred;
subplot(1,2,1); showIm(fine0, 'auto2', 'auto', 'fine0');

%% The blurred and fine images contain all the information found in
%% the original image.  Trivially, adding the blurred image to the
%% fine scale detail will reconstruct the original.  We can compare
%% the original image to the sum of blurred and fine using the
%% "imStats" function, which reports on the statistics of the
%% difference between it's arguments:
imStats(im, blurred+fine0);

%% Since the filter is a lowpass filter, we might want to subsample
%% the blurred image (e.g., convert the 512 by 512 to 256 by 256).  
%% This may cause some aliasing (depends on the
%% filter), but the decomposition structure given above will still be
%% possible.  The corrDn function correlates (same as convolution, but
%% flipped filter) and downsamples in a single operation (for
%% efficiency).  The string 'reflect1' tells the function to handle
%% boundaries by reflecting the image about the edge pixels.  Notice
%% that the blurred1 image is half the size (in each dimension) of the
%% original image.
lo_filt = 2*binom5*binom5';  
blurred1 = corrDn(im,lo_filt,'reflect1',[2 2]);
subplot(1,2,2); showIm(blurred1,'auto2','auto','blurred1');

%% Now, to extract fine scale detail, we must interpolate the image
%% back up to full size before subtracting it from the original.  The
%% upConv function does upsampling (padding with zeros between
%% samples) followed by convolution.  This can be done using the
%% lowpass filter that was applied before subsampling or it can be
%% done with a different filter.
fine1 = im - upConv(blurred1,lo_filt,'reflect1',[2 2],[1 1],size(im));
subplot(1,2,1); showIm(fine1,'auto2','auto','fine1');

%% We now have a technique that takes an image, computes two new
%% images (blurred1 and fine1) containing the coarse scale information
%% and the fine scale information.  We can also (trivially)
%% reconstruct the original from these two (even if the subsampling of
%% the blurred1 image caused aliasing):

recon = fine1 + upConv(blurred1,lo_filt,'reflect1',[2 2],[1 1],size(im));
imStats(im, recon);

%% The examples above are part of what is known as a 
%% Laplacian pyramid, but for our purposes we would like
%% to use pyramids with oriented filters. We use a steerable
%% pyramid.

%% Steerable pyramid:
%% To construct a steerable pyramid: The image is first 
%% divided into a highpass and lowpass subband. The highpass
%% is considered a residual. The lowpass band is split into
%% a low(er)pass band and k+1 oriented filters. The pyramid
%% is built by each time splitting the lowpass band into a
%% low(er)pass band and k+1 oriented filters. We therefore
%% have k+1 oriented filters at each level of the pyramid. We
%% are left also with a lowest pass residual. 

%% build a Steerable pyramid:

numlevs = 3;
numoris = 3; % pyramid is built with k+1 (here 4) orientations
[pyr,pind] = buildSFpyr(im, numlevs, numoris);

%% pind is an index into the sizes of the orientation bands
%% at each level of the pyramid.
pind

%% The first term 256 256 represents the highpass residual
%% which has the same dimensionality as the image size
size(im)

%% The next 4 terms 256 256 represent the oriented filters
%% at the highest level of the pyramid
%% The 4 terms 128 128 represent the next 4 oriented
%% filters at the next level of the pyramid. And so on.

%% pyr contains all the filter outputs of the pyramid.
size(pyr)

%% There are built in functions for looking at the subbands
%% Look at first (vertical) bands, different scales:
for s = 1:min(4,spyrHt(pind))
  band = spyrBand(pyr,pind,s,1);
  subplot(2,2,s); showIm(band);
end

%% look at all orientation bands at one level (scale):
for b = 1:spyrNumBands(pind)
  band = spyrBand(pyr,pind,1,b);
  subplot(2,2,b);
  showIm(band);
end

%% To access the high-pass and low-pass bands:
clf;
low = pyrLow(pyr,pind);
showIm(low);
high = spyrHigh(pyr,pind);
showIm(high);

%% Display the whole pyramid (except for the highpass residual band),
%% with images shown at proper relative sizes:
showSpyr(pyr,pind);

%% Spin a level of the pyramid, interpolating (steering to)
%% intermediate orienations:

%% Choose a filter set (options are 'sp0Filters', 'sp1Filters',
%% 'sp3Filters', 'sp5Filters'):
filts = 'sp3Filters';           % filter set with 3+1 orientations
[lo0filt,hi0filt,lofilt,bfilts,steermtx,harmonics] = eval(filts);
[lev,lind] = spyrLev(pyr,pind,2);
lev2 = reshape(lev,prod(lind(1,:)),size(bfilts,2));
figure(1); subplot(1,1,1); showIm(spyrBand(pyr,pind,2,1));
M = moviein(16);
for frame = 1:16
  steered_im = steer(lev2, 2*pi*(frame-1)/16, harmonics, steermtx);
  showIm(reshape(steered_im, lind(1,:)),'auto2');
  M(:,frame) = getframe;
end

%% Reconstruct. 
res = reconSFpyr(pyr, pind);
showIm(im + i * res);
imStats(im,res);

%% We'd like to also look at the filters themselves, which are used
%% in the transformation of the image into subbands.

%%% *To do*: We can do this by constructing a pyramid in which
%%% the input image is an impulse (zero everywhere except the
%%% middle location). This process describes how much each sample
%%% of the input contributes to each of the subbands. Construct
%%% a pyramid with an impulse image and look at a subband
%%% at the first level of the pyramid and the first orientation.
%%% Try other examples. You can go back to some of the code
%%% above but with the impulse image instead of Einstein...

pyrheight = 3; orient_order = 3;
im_impulse = zeros(size(im));
thelen = round(length(im)/2);
im_impulse(thelen,thelen)=1;
[pyr,pind] = buildSFpyr(im_impulse,pyrheight, orient_order);
band = spyrBand(pyr,pind,1,1);
showIm(band(thelen-10:thelen+10,thelen-10:thelen+10))

%% You can try different subbands (set the lengths accordingly).

%% We can also look at the inverse, of what filter goes from a
%% subband back to the image. We do this by placing an impulse 
%% in a given subband, and reconstructing.

orient_order=3;
pyrheight=3;
theLev=2; theOri=1;
bnum = 1+theOri+(theLev-1)*(orient_order+1)
[pyr,pind] = buildSFpyr(zeros(60),pyrheight, orient_order);
bandDims = pind(bnum,:);
bandCtr = floor(bandDims/2)+1;
bandImCtr = (bandCtr-1+[(mod(theOri,2)==1), (theOri>=2)]/2)*2^theLev + 1;
bandImCtr= floor(bandImCtr/2);
bandCtrInd = (bandCtr(2)-1)*bandDims(1) + bandCtr(1);
bandIndices = pyrBandIndices(pind,bnum);
pyr(bandIndices(1) -1 + bandCtrInd) = 1;
clf; showIm(spyrBand(pyr,pind,theLev,theOri));  % check:  band is an impulse
res = reconSFpyr(pyr,pind);
showIm(res(bandImCtr(1)-10:bandImCtr(1)+10, bandImCtr(2)-10:bandImCtr(2)+10));

%%% If you have time at the end, and want to learn more about pyramids,
%%% you can look at the full tutorial in:
%%% matlabPyrTools/TUTORIALS/pyramids.m


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% (2) Optimized filters from images

%%% To optimize from scenes, we will work with a fixed filter
%%% size of 12 by 12. Rather than looking at a full image, we
%%% will consider many samples of 12 by 12 patches from the image.

%%% We will start out by taking many samples of random
%%% patches from an image, and then do a linear transformation
%%% (e.g., PCA, ICA) on the patches.

%%% Read in an image -- try different examples
multFact = 256;
im=vanRead('imk00649.iml');
im = multFact*(im./max(max(im)));
im = im(1:512,1:512);
size(im)
showIm(im)

%%% *To do": Write a Matlab function
%%% sig = get_patches(im, patch_dim, num_patches)
%%% that takes num_patches random patches from an
%%% input image and saves each patch as a column
%%% of the output matrix sig.
patch_dim = 12;
num_patches = 10000;
sig = get_patches(im, patch_dim, num_patches);
size(sig)

%%%% *To do*: Look at example random patches in sig by reshaping
%%%% the columns of the matrix to 12 by 12
showIm(reshape(sig(:,round(rand*num_patches)),patch_dim,patch_dim));
 
%%% PCA:
%%% *To do *: Consider linear transformations on the image patch ensemble
%%% of the form: newsig = mat * sig        (where * is multiplication)
%%% First compute a PCA transform of the patches sig.
%%% You can use the function pca, in which you input sig and
%%% obtain as the output newsig, mat (and also the eigenvectors
%%% and eigenvalues). Even better, you can write the pca yourself.
%%% First, do an eigenvector decomposition of sig*sig'./numsigs,
%%% obtaining eigenvalues D and eigenvectors V.
%%% Note that: sig sig' = V D V' with V orthogonal
%%% V' sig sig' V = D; newsig newsig' = D, where newsig = V' sig
%%% Next, sort the eigenvalues and eigenvectors from largest to smallest.
%%% Next, whiten by multiplying the non-zero eigenvalues D by sqrt(1./D). 
%%% Compute the mat matrix as the new D values times V'.
%%% Then compute newsig as mat times sig.

[newsig, evecs, evals, mat]=mypca(sig);
%% Note: this is a pca function in new Matlab different 
%% from one in matlabPyrTools
% manually
% pca
numsigs = size(sig,2);
[V, D] = eig(sig*sig'./numsigs);
D=diag(D);
D=D(length(D):-1:1);
V=V(:,length(D):-1:1);
mat = V;
newsig = mat*sig;
nonZeroInd = find(abs(D) > eps);
% whiten
whitener = diag(sqrt(1./D(nonZeroInd))');
mat=whitener*V';
newsig = mat*sig;

%%% *To do*: Confirm that the transform produced newsig that are
%%% uncorrelated.
showIm(newsig*newsig')


%%% *To do*: Look at the sizes of sig, newsig, and mat.
%%% Plot the eigenvalues.
%%% Look at the principle components (eigenvectors):
%%% The rows of the matrix mat are called projection functions.
%%% They project from the input columns (patches of image)
%%% onto the principle components.
%%% We can view these by reshaping each row (mat(i,:)) to 12 by 12
%%% What do the eigenvectors look like for high eigenvalues? For low ones?
plot(evals, 'o');
showIm(reshape(mat(round(rand*patch_dim^2),:),patch_dim,patch_dim))
showIm(reshape(mat(1,:),patch_dim,patch_dim))         
showIm(reshape(mat(5,:),patch_dim,patch_dim))         
showIm(reshape(mat(10,:),patch_dim,patch_dim))         

%%% *To do*: We are also interested in the transform:
%%% sig = mat_inverse * newsig
%%% This is a transform from the output of pca back to the
%%% original image patches. The matrix mat_inverse is the
%%% inverse of mat. The columns of the matrix mat_inverse
%%% are known as basis functions. The input is a linear
%%% combination of the columns of mat_inverse.
%%% To view the basis functions, we need to view columns of mat_inverse
%%% using matlab's inv(mat). Look at different columns of mat_inverse.
mat_inverse = inv(mat); figure(1);
showIm(reshape(mat_inverse(:,round(rand*patch_dim^2)),patch_dim,patch_dim))
showIm(reshape(mat_inverse(:,1),patch_dim,patch_dim))             
showIm(reshape(mat_inverse(:,10),patch_dim,patch_dim))             

%%% Since pca is an orthonormal transform mat*mat_inverse is a diagonal
%%% matrix. Note that if we use pca without whitening then the basis
%%% functions are exactly given by the columns of the transpose
%%% of mat since the inverse is equal to the transpose.
showIm(mat*mat_inverse)

%%% We will now look at ICA.
%%% There are several packages available on the web for ICA.

%%% There are various versions of ICA available:
%%% Bell and Sejnowski ICA:
%%% http://cnl.salk.edu/~tony/ica.html
%%% Hoyer and Hyvarinen fast ICA applied to images (imageica):
%%% http://www.cs.helsinki.fi/u/phoyer/software.html
%%% And sparse coding:
%%% Olshausen and Field:
%%% http://redwood.berkeley.edu/bruno/sparsenet/

%%% We will use the fastICA code of Hoyer and Hyvarinen
%%% applied to images
%%% http://www.cs.helsinki.fi/u/phoyer/software.html
%%% cd /Users/schwartzlab/Desktop/BerkeleyCourse2011/Lab1
%%% path(path,'/Volumes/datahd/Desktop/BerkeleyCourse2011/Lab1/imageica')
%%% path(path,'/Volumes/datahd/Desktop/BerkeleyCourse2011/Lab1/imageica/code')
%%% path(path,'/Volumes/datahd/Desktop/BerkeleyCourse2011/Lab1/imageica/results')

sig = get_patches(im, patch_dim, num_patches);
clear X;
global X;
X = sig;

% Subtract local mean from each patch
%X = X-ones(size(X,1),1)*mean(X);

if 0
% Pre Whiten
[newsig, evecs, evals, mat]=pca(X);
whiteningMatrix = mat;
dewhiteningMatrix = inv(mat);
X = newsig;
end

% Pre Whiten
% Use this one
covarianceMatrix = X*X'/size(X,2);
[E, D] = eig(covarianceMatrix);
[dummy,order] = sort(diag(-D));
%order=order:-1:1;
% to make it same as above code
E = E(:,order);
d = diag(D);
d = real(d.^(-0.5));
D = diag(d(order));
X = D*E'*X;
whiteningMatrix = D*E';
dewhiteningMatrix = E*D^(-1);

p.seed = 1;
p.write = 20;
p.model = 'ica';
p.algorithm = 'fixed-point';
p.components = min(size(X));
numIters = 500;            % run for 1000 trials
estimateModif( whiteningMatrix, dewhiteningMatrix, 'tryica.mat', p, numIters );
% Original function is estimate.m which runs infinitely; 
% estimateModif.m runs for numIters times

% Old
% estimate( whiteningMatrix, dewhiteningMatrix, 'tryica.mat', p );
% This runs infinitely... hit control c to stop after around 1000 trials

% Load saved results
load('tryica.mat');

% *To do*: The columns of A contain the basis functions; view them...
thecol=round(rand*144);
showIm(reshape(A(:,thecol),patch_dim,patch_dim))

%%% Not included
%%% We can also use an analytical function based on Cardoso
%%% ica4. Do this tranform and look at the projection and basis functions.
%%[ica_result, mat]=ica4(sig);
%%basis=inv(mat);
%%showIm(reshape(basis(:,round(rand*patch_dim^2)),patch_dim,patch_dim))
