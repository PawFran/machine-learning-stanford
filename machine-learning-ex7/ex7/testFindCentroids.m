X = reshape(sin(1:50), 10, 5);
cent = X(7:10,:);
idx = findClosestCentroids(X, cent)
