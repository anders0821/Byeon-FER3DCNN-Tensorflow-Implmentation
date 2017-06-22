clc;
clear;
close all;

% load XY3_EYELIP_NOBG_INTERP_MIRROR.mat
load FEEDTUMFINETUNE2_XY3_EYELIP_NOBG_INTERP_MIRROR.mat

N_TRAIN = size(TRAINX, 6)
TRAINX = single(reshape(TRAINX, [1 128 128 16 N_TRAIN]));
TRAINY = single(reshape(TRAINY, [6 N_TRAIN]));
TRAINX_NEW = zeros([1 64 48 5 N_TRAIN], 'single');
for i=1:N_TRAIN
    im = squeeze(TRAINX(1, :, :, :, i));
    % figure
    % imshow(squeeze(im(:,:,1)));
    % figure
    % imshow(squeeze(im(:,:,end)));
    im = imresize3(im, [64, 48, 5]);
    % figure
    % imshow(squeeze(im(:,:,1)));
    % figure
    % imshow(squeeze(im(:,:,end)));
    % return
    
    TRAINX_NEW(1, :, :, :, i) = im;
end
TRAINX = TRAINX_NEW;

N_VAL = size(VALX, 6)
VALX = single(reshape(VALX, [1 128 128 16 N_VAL]));
VALY = single(reshape(VALY, [6 N_VAL]));
VALX_NEW = zeros([1 64 48 5 N_VAL], 'single');
for i=1:N_VAL
    im = squeeze(VALX(1, :, :, :, i));
    % figure
    % imshow(squeeze(im(:,:,1)));
    % figure
    % imshow(squeeze(im(:,:,end)));
    im = imresize3(im, [64, 48, 5]);
    % figure
    % imshow(squeeze(im(:,:,1)));
    % figure
    % imshow(squeeze(im(:,:,end)));
    % return
    
    VALX_NEW(1, :, :, :, i) = im;
end
VALX = VALX_NEW;

N_TEST = size(TESTX, 6)
TESTX = single(reshape(TESTX, [1 128 128 16 N_TEST]));
TESTY = single(reshape(TESTY, [6 N_TEST]));
TESTX_NEW = zeros([1 64 48 5 N_TEST], 'single');
for i=1:N_TEST
    im = squeeze(TESTX(1, :, :, :, i));
    % figure
    % imshow(squeeze(im(:,:,1)));
    % figure
    % imshow(squeeze(im(:,:,end)));
    im = imresize3(im, [64, 48, 5]);
    % figure
    % imshow(squeeze(im(:,:,1)));
    % figure
    % imshow(squeeze(im(:,:,end)));
    % return
    
    TESTX_NEW(1, :, :, :, i) = im;
end
TESTX = TESTX_NEW;

save CONVERT.mat TRAINX VALX TESTX TRAINY VALY TESTY
