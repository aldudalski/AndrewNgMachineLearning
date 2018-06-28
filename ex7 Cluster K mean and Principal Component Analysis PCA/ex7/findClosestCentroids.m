function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.



% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X,1);
idx = zeros(m, 1);

% loop over every example
for i = 1:m
  %iterate over every
  min_norm = Inf; % track the minimum norm
  min_norm_j = -1; % track the smallest norm - make wrong initially.
  for j = 1:K
    % calculate the norm
    norm = sum((X(i,:)' - centroids(j,:)').^2);
    if norm < min_norm
      min_norm = norm;
      min_norm_j = j;
    end
  end
  if min_norm_j == -1 disp('ERROR:   MIN NORM J IS -1 no index selected') end; %should never happen
  idx(i)=min_norm_j;
end

% =============================================================

end
