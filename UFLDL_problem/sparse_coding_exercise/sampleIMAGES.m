function patches = sampleIMAGES(IMAGES,patchsize,numpatches)
% sampleIMAGES
% Returns 10000 patches for training



% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1
% for i=1:numpatches
%     a=unidrnd(10);
%     b=unidrnd(497);
%     c=b+15;
%     patches(:,i)=reshape(images(b:c,b:c,a)',[],1);
% end;
tic
image_size=size(IMAGES);
i=randi(image_size(1)-patchsize+1,1,numpatches);%生成元素值随机为大于0且小于image_size(1)-patchsize+1的1行numpatches矩阵
j=randi(image_size(2)-patchsize+1,1,numpatches);
k=randi(image_size(3),1,numpatches);
for num=1:numpatches
        patches(:,num)=reshape(IMAGES(i(num):i(num)+patchsize-1,j(num):j(num)+patchsize-1,k(num)),1,patchsize*patchsize);
end
toc








%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
%patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
