function [im]=imresize3(im, dims)
    assert(~any(any(any(isnan(im)))));
    [x, y, z] = ndgrid(linspace(1,size(im,1), dims(1)), linspace(1,size(im,2), dims(2)), linspace(1,size(im,3), dims(3)));
    im = interp3(im,y,x,z);
    assert(~any(any(any(isnan(im)))));
end
