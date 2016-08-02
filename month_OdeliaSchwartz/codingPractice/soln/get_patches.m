function patches = get_patches(im, patch_dim, num_patches)
% pick random starting x and y positions in im, and take patch
% as column vector

patches = [];
patches = zeros(patch_dim^2, num_patches);

for i = 1:num_patches
   thex = 0; they = 0;
   while (thex < 1) | thex > (size(im,2)-patch_dim)
      thex = round(rand*size(im,2));
   end
   while (they < 1) | they > (size(im,1)-patch_dim)
      they = round(rand*size(im,1));
   end
   the_patch = im(they:they+patch_dim-1, thex:thex+patch_dim-1);
   %patches = [patches, the_patch(:)];
   patches(:,i) = the_patch(:);
end