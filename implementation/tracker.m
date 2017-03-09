function results = tracker(params)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%parameters
search_area_scale = params.search_area_scale;
output_sigma_factor = params.output_sigma_factor;
refinement_iterations = params.refinement_iterations;
max_image_sample_size = params.max_image_sample_size;
min_image_sample_size = params.min_image_sample_size;
nScales = params.number_of_scales;
scale_step = params.scale_step;
CG_tol = params.CG_tol;
newton_iterations = params.newton_iterations;
features = params.t_features;

% Set some default parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
if ~isfield(params, 'interpolation_method')
    params.interpolation_method = 'none';
end
if ~isfield(params, 'interpolation_centering')
    params.interpolation_centering = false;
end
if ~isfield(params, 'interpolation_windowing')
    params.interpolation_windowing = false;
end
if ~isfield(params, 'clamp_position')
    params.clamp_position = false;
end

s_frames = params.s_frames;
pos = floor(params.init_pos(:)');
target_sz = floor(params.wsize(:)');

debug = params.debug;
visualization = params.visualization || debug;

num_frames = numel(s_frames);

params.nSamples = min(params.nSamples, num_frames);

%notation: variables ending with f are in the frequency domain.

init_target_sz = target_sz;

% Calculate feature dimension
im = imread(s_frames{1});
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% use the maximum feature ratio right now todetermin the search size

search_area = prod(init_target_sz * search_area_scale);

if search_area > max_image_sample_size
    currentScaleFactor = sqrt(search_area / max_image_sample_size);
elseif search_area < min_image_sample_size
    currentScaleFactor = sqrt(search_area / min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

%window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor( base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*4]; % for testing
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');

% Set feature info
img_sample_sz = feature_info.img_sample_sz;
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
feature_dim = feature_info.dim;
num_feature_blocks = length(feature_dim);
feature_reg = permute(num2cell(feature_info.penalty), [2 3 1]);

% Size of the extracted feature maps
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% Number of Fourier coefficients to save for each filter layer. This will
% be an odd number.
filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size.
output_sz = max(filter_sz, [], 1);

% How much each feature block has to be padded to the obtain output_sz
pad_sz = cellfun(@(filter_sz) (output_sz - filter_sz) / 2, filter_sz_cell, 'uniformoutput', false);

% Compute the Fourier series indices and their transposes
ky = circshift(-floor((output_sz(1) - 1)/2) : ceil((output_sz(1) - 1)/2), [1, -floor((output_sz(1) - 1)/2)])';
kx = circshift(-floor((output_sz(2) - 1)/2) : ceil((output_sz(2) - 1)/2), [1, -floor((output_sz(2) - 1)/2)]);
ky_tp = ky';
kx_tp = kx';

% construct the Gaussian label function using Poisson summation formula
% sig_y = sqrt(prod(floor(base_target_sz))) * output_sigma_factor * (output_sz ./ img_sample_sz);
sig_y = sqrt(prod(floor(base_target_sz))) * output_sigma_factor * (output_sz ./ img_support_sz);
yf_y = single(sqrt(2*pi) * sig_y(1) / output_sz(1) * exp(-2 * (pi * sig_y(1) * ky / output_sz(1)).^2));
yf_x = single(sqrt(2*pi) * sig_y(2) / output_sz(2) * exp(-2 * (pi * sig_y(2) * kx / output_sz(2)).^2));
y_dft = yf_y * yf_x;

% Compute the labelfunction at the filter sizes
yf = cellfun(@(sz) fftshift(resizeDFT2(y_dft, sz, false)), filter_sz_cell, 'uniformoutput', false);
yf = compact_fourier_coeff(yf);

% construct cosine window
cos_window = cellfun(@(sz) single(hann(sz(1)+2)*hann(sz(2)+2)'), feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cos_window(2:end-1,2:end-1), cos_window, 'uniformoutput', false);

% Compute Fourier series of interpolation function
[interp1_fs, interp2_fs] = cellfun(@(sz) get_interp_fourier(sz, params), filter_sz_cell, 'uniformoutput', false);

% Get the reg_window_edge parameter
reg_window_edge = {};
for k = 1:length(features)
    if isfield(features{k}.fparams, 'reg_window_edge')
        reg_window_edge = cat(3, reg_window_edge, permute(num2cell(features{k}.fparams.reg_window_edge(:)), [2 3 1]));
    else
        reg_window_edge = cat(3, reg_window_edge, cell(1, 1, length(features{k}.fparams.nDim)));
    end
end

% Construct spatial regularization filter
reg_filter = cellfun(@(reg_window_edge) get_reg_filter(img_support_sz, base_target_sz, params, reg_window_edge), reg_window_edge, 'uniformoutput', false);

% Compute the energy of the filter (used for preconditioner)
reg_energy = cellfun(@(reg_filter) real(reg_filter(:)' * reg_filter(:)), reg_filter, 'uniformoutput', false);

if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    
    scaleFactors = scale_step .^ scale_exp;
    
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

% initialize the projection matrix
rect_position = zeros(num_frames, 4);

time = 0;

% Initialize and allocate
prior_weights = [];
sample_weights = [];
latest_ind = [];
samplesf = cell(1, 1, num_feature_blocks);
for k = 1:num_feature_blocks
    samplesf{k} = complex(zeros(params.nSamples,feature_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2,'single'));
end

residuals_pcg = [];

for frame = 1:num_frames,
    %load image
    im = imread(s_frames{frame});
    if size(im,3) > 1 && is_color_image == false
        im = im(:,:,1);
    end

    tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we 
    % just want to initialize the tracker there
    if frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            xt = extract_features(im, pos, currentScaleFactor*scaleFactors, features, global_fparams);
            
            % Do windowing of features
            xt = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xtf = cellfun(@cfft2, xt, 'uniformoutput', false);
            
            % Interpolate features to the continuous domain
            xtf = interpolate_dft(xtf, interp1_fs, interp2_fs);
            
            % Compute convolution for each feature block in the Fourier domain
            scores_fs_feat = cellfun(@(hf, xf, pad_sz) padarray(sum(bsxfun(@times, hf, xf), 3), pad_sz), hf_full, xtf, pad_sz, 'uniformoutput', false);
            
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = ifftshift(ifftshift(permute(sum(cell2mat(scores_fs_feat), 3), [1 2 4 3]), 1), 2);
            
            % Optimize the continuous score function with Newton's method.
            [trans_row, trans_col, scale_ind] = optimize_scores(scores_fs, newton_iterations, ky_tp, kx_tp);
            
            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = round([trans_row, trans_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(scale_ind));

            % set the scale
            currentScaleFactor = currentScaleFactor * scaleFactors(scale_ind);
            % adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            % update position
            old_pos = pos;
            pos = pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            iter = iter + 1;
        end
        
        % debug visualization
%         if debug
%             figure(101);
%             subplot_cols = ceil(sqrt(nScales));
%             subplot_rows = ceil(nScales/subplot_cols);
%             for scale_ind = 1:nScales
%                 subplot(subplot_rows,subplot_cols,scale_ind);
%                 imagesc(fftshift(response(:,:,scale_ind)));colorbar; axis image;
%                 title(sprintf('Scale %i,  max(response) = %f', scale_ind, max(max(response(:,:,scale_ind)))));
%             end
%         end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Update the weights
    [prior_weights, replace_ind] = update_prior_weights(prior_weights, sample_weights, latest_ind, frame, params);
    latest_ind = replace_ind;
    sample_weights = prior_weights;
    
    % Extract image region for training sample
    xl = extract_features(im, pos, currentScaleFactor, features, global_fparams);
    
    % Do windowing of features
    xl = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
    
    % Compute the fourier series
    xlf = cellfun(@cfft2, xl, 'uniformoutput', false);
    
    % Interpolate features to the continuous domain
    xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);
    
    % New sample to be added
    xlf = cellfun(@(xf) xf(:,1:(size(xf,2)+1)/2,:), xlf, 'uniformoutput', false);
    
    % Insert the new training sample
    for k = 1:num_feature_blocks
        samplesf{k}(replace_ind,:,:,:) = permute(xlf{k}, [4 3 1 2]);
    end
    
    % Construct the right hand side vector
    rhs_samplef = cellfun(@(xf) permute(mtimesx(sample_weights, 'T', xf, 'speed'), [3 4 2 1]), samplesf, 'uniformoutput', false);
    rhs_samplef = cellfun(@(xf, yf) bsxfun(@times, conj(xf), yf), rhs_samplef, yf, 'uniformoutput', false);
    
    new_sample_energy = cellfun(@(xlf) abs(xlf .* conj(xlf)), xlf, 'uniformoutput', false);
    
    if frame == 1
        % Initialize the filter
        hf = cell(1,1,num_feature_blocks);
        for k = 1:num_feature_blocks
            hf{k} = complex(zeros([filter_sz(k,1) (filter_sz(k,2)+1)/2 feature_dim(k)], 'single'));
        end
        
        % Initialize Conjugate Gradient parameters
        p = [];
        rho = [];
        max_CG_iter = params.init_max_CG_iter;
        sample_energy = new_sample_energy;
    else
        max_CG_iter = params.max_CG_iter;
        
        if params.CG_forgetting_rate == inf || params.learning_rate >= 1
            % CG will be restarted
            p = [];
            rho = [];
        else
            rho = rho / (1-params.learning_rate)^params.CG_forgetting_rate;
        end
        
        % Update the approximate average sample energy using the learning
        % rate. This is only used to construct the preconditioner.
        sample_energy = cellfun(@(se, nse) (1 - params.learning_rate) * se + params.learning_rate * nse, sample_energy, new_sample_energy, 'uniformoutput', false);
    end
    
    % Construct preconditioner
    diag_M = cellfun(@(m, reg_energy) (1-params.precond_reg_param) * bsxfun(@plus, params.precond_data_param * m, (1-params.precond_data_param) * mean(m,3)) + params.precond_reg_param*reg_energy, sample_energy, reg_energy, 'uniformoutput',false);
    
    % do conjugate gradient
    [hf, flag, relres, iter, res_norms, p, rho] = pcg_ccot(...
        @(x) lhs_operation(x, samplesf, reg_filter, sample_weights, feature_reg),...
        rhs_samplef, CG_tol, max_CG_iter, ...
        @(x) diag_precond(x, diag_M), ...
        [], hf, p, rho);
    
    % Make the filter symmetric (avoid roundoff errors)
    hf = symmetrize_filter(hf);

    % Reconstruct the full Fourier series
    hf_full = full_fourier_coeff(hf);
    
    % Update the target size (only used for computing output box)
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    %save position and calculate FPS
    rect_position(frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    
    time = time + toc();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % debug visualization
    if debug
        figure(20)
%         set(gcf,'units','normalized','outerposition',[0 0 1 1]);
        subplot_cols = num_feature_blocks;
        subplot_rows = 3;%ceil(feature_dim/subplot_cols);
        for disp_layer = 1:num_feature_blocks;
            subplot(subplot_rows,subplot_cols,disp_layer);
            imagesc(prod(filter_sz(disp_layer,:)) * mean(abs(ifft2(conj(ifftshift(hf_full{disp_layer})), 'symmetric')), 3)); 
            colorbar;
            axis image;
            subplot(subplot_rows,subplot_cols,disp_layer+subplot_cols);
            imagesc(mean(abs(xl{disp_layer}), 3)./(cos_window{disp_layer}+eps)); 
            colorbar;
            axis image;
            if frame > 1
                subplot(subplot_rows,subplot_cols,disp_layer+2*subplot_cols);
                imagesc(prod(output_sz) * fftshift(ifft2(ifftshift(scores_fs_feat{disp_layer}(:,:,1,scale_ind)), 'symmetric')));
                colorbar;
                axis image;
            end
        end
        
%         figure(102);plot(sample_weights);title('Sample weights');
        
        residuals_pcg = [residuals_pcg; res_norms/norm_cdcf(rhs_samplef)];
        res_start_ind = max(1, length(residuals_pcg)-40*max_CG_iter);
        figure(99);plot(res_start_ind:length(residuals_pcg), residuals_pcg(res_start_ind:end));
        axis([res_start_ind, length(residuals_pcg), 0, min(max(residuals_pcg(res_start_ind:end)), 0.2)]);
    end
    
    %visualization
    if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if frame == 1,  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking');
%             set(fig_handle, 'Position', [100, 100, size(im,2), size(im,1)]);
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
            
%             output_name = 'Regularized';
%             opengl software;
%             writer = VideoWriter(output_name, 'MPEG-4');
%             writer.FrameRate = 5;
%             open(writer);
        else
            % Do visualization of the sampled confidence scores overlayed
            resp_sz = round(img_support_sz*currentScaleFactor*scaleFactors(scale_ind));
            xs = floor(old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            
            % To visualize the continuous scores, sample them 10 times more
            % dense than output_sz. This is implemented an an ifft2.
            % First pad the fourier series with zeros.
            sampled_scores_display_dft = resizeDFT2(scores_fs(:,:,scale_ind), 10*output_sz, false);
            % Then do inverse DFT and rescale correctly
            sampled_scores_display = fftshift(prod(10*output_sz) * ifft2(sampled_scores_display_dft, 'symmetric'));
            
            figure(fig_handle);
%                 set(fig_handle, 'Position', [100, 100, 100+size(im,2), 100+size(im,1)]);
            imagesc(im_to_show);
            hold on;
            resp_handle = imagesc(xs, ys, sampled_scores_display); colormap hsv;
            alpha(resp_handle, 0.5);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
            
%                 axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        end
        
        drawnow
%         if frame > 1
%             if frame < inf
%                 writeVideo(writer, getframe(gcf));
%             else
%                 close(writer);
%             end
%         end
         %pause
    end
end

% close(writer);

fps = numel(s_frames) / time;

disp(['fps: ' num2str(fps)])

results.type = 'rect';
results.res = rect_position;%each row is a rectangle
results.fps = fps;
