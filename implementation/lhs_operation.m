function hf_out = lhs_operation(hf, samplesf, reg_filter, sample_weights, feature_reg)

% This is the left-hand-side operation in Conjugate Gradient

% size of the padding
num_features = length(hf);
output_sz = [size(hf{1},1), 2*size(hf{1},2)-1];
pad_sz = cellfun(@(hf) (output_sz - [size(hf,1), 2*size(hf,2)-1]) / 2, hf, 'uniformoutput',false);

% Compute the operation corresponding to the data term in the optimization
% (blockwise matrix multiplications)
%implements: A' diag(sample_weights) A f

% sum over all features in each block
sh_cell = cellfun(@(hf,samplesf) mtimesx(samplesf, permute(hf, [3 4 1 2]), 'speed'), hf, samplesf, 'uniformoutput', false);

% sum over all feature blocks
sh = sh_cell{1};    % assumes the feature with the highest resolution is first
for k = 2:num_features
    sh(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) = ...
        sh(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) + sh_cell{k};
end

% weight all the samples
sh = bsxfun(@times,sample_weights,sh);

% multiply with the transpose
hf_out = cellfun(@(samplesf,pad_sz) permute(conj(mtimesx(sh(:,1,1+pad_sz(1):end-pad_sz(1), 1+pad_sz(2):end), 'C', samplesf, 'speed')), [3 4 2 1]), ...
    samplesf, pad_sz, 'uniformoutput', false);

% compute the operation corresponding to the regularization term (convolve
% each feature dimension with the DFT of w, and the tramsposed operation)
% add the regularization part

reg_pad = cellfun(@(hf, reg_filter) min(size(reg_filter,2)-1, size(hf,2)-1), hf, reg_filter, 'uniformoutput', false);

% add part needed for convolution
hf_conv = cellfun(@(hf,reg_pad) cat(2, hf, conj(rot90(hf(:, end-reg_pad:end-1, :), 2))), hf, reg_pad, 'uniformoutput', false);

% do first convolution
hf_conv = cellfun(@(hf_conv, reg_filter) convn(hf_conv, reg_filter), hf_conv, reg_filter, 'uniformoutput', false); 

% do final convolution and put toghether result
hf_out = cellfun(@(hf_conv, hf_data, reg_filter, hf, reg_pad, feature_reg) hf_data + convn(hf_conv(:,1:end-reg_pad,:), reg_filter, 'valid') + feature_reg * hf, ...
    hf_conv, hf_out, reg_filter, hf, reg_pad, feature_reg, 'uniformoutput', false); 

end