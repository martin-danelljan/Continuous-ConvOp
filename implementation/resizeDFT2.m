function resizeddft  = resizeDFT2(inputdft, desiredSize, rescale)

% Resize the DFT of a sample by either removing the highest frequencies or
% padding with zeroes. This corresponds to interpolation with complex
% exponentials in the spatial domain.

[imh, imw, n1, n2] = size(inputdft);
imsz = [imh, imw];

if any(desiredSize ~= imsz)
    minsz = min(imsz, desiredSize);
    
    % Do rescaling as default
    if nargin < 3 || rescale
        scaling = prod(desiredSize)/prod(imsz);
    else
        scaling = 1;
    end
    
    resizeddft = complex(zeros([desiredSize n1 n2], 'single'));
    
    mids = ceil(minsz/2);
    mide = floor((minsz-1)/2) - 1;
    
    resizeddft(1:mids(1), 1:mids(2), :, :) = scaling * inputdft(1:mids(1), 1:mids(2), :, :);
    resizeddft(1:mids(1), end - mide(2):end, :, :) = scaling * inputdft(1:mids(1), end - mide(2):end, :, :);
    resizeddft(end - mide(1):end, 1:mids(2), :, :) = scaling * inputdft(end - mide(1):end, 1:mids(2), :, :);
    resizeddft(end - mide(1):end, end - mide(2):end, :, :) = scaling * inputdft(end - mide(1):end, end - mide(2):end, :, :);
else
    resizeddft = inputdft;
end
end