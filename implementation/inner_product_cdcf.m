function ip = inner_product_cdcf(xf, yf)

% Computes the inner product between two filters.

ip_cell = cellfun(@(xf, yf) real(2*(xf(:)' * yf(:)) - reshape(xf(:,end,:), [], 1, 1)' * reshape(yf(:,end,:), [], 1, 1)), xf, yf, 'uniformoutput', false');
ip = sum(cell2mat(ip_cell));

% ip_cell = cellfun(@(xf, yf) real(xf(:)' * yf(:)), xf, yf, 'uniformoutput', false');
% ip = sum(cell2mat(ip_cell));