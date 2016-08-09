function feature_map = get_cnn_layers(im, fparams, gparams)

% Get layers from a cnn.

if size(im,3) == 1
    im = repmat(im, [1 1 3]);
end

im_sample_size = size(im);

%preprocess the image
if ~isequal(im_sample_size(1:2), fparams.net.meta.normalization.imageSize(1:2))
    im = imresize(single(im), fparams.net.meta.normalization.imageSize(1:2));
else
    im = single(im);
end

% Normalize with average image
im = bsxfun(@minus, im, fparams.net.meta.normalization.averageImage);

cnn_feat = vl_simplenn(fparams.net, im);

feature_map = cell(1,1,length(fparams.output_layer));

for k = 1:length(fparams.output_layer)
    if fparams.downsample_factor(k) == 1
        feature_map{k} = cnn_feat(fparams.output_layer(k) + 1).x(fparams.start_ind(k,1):fparams.end_ind(k,1), fparams.start_ind(k,2):fparams.end_ind(k,2), :, :);
    else
        feature_map{k} = average_feature_region(cnn_feat(fparams.output_layer(k) + 1).x(fparams.start_ind(k,1):fparams.end_ind(k,1), fparams.start_ind(k,2):fparams.end_ind(k,2), :, :), fparams.downsample_factor(k));
    end
end

