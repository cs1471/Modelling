
% IM = vanRead( FILENAME )
%
% Load a van hateren image into a MatLab matrix.
% This format is 16 bit grayscale (raw).
% Feb 2001

function im = vanRead( fname );

fid=fopen(fname, 'r', 'ieee-be');
im=fread(fid, [1536, 1024], 'int16');
im=im';
