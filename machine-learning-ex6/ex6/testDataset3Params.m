x1plot = linspace(-2, 2, 10)';
x2plot = linspace(-2, 2, 10)';
[X1, X2] = meshgrid(x1plot, x2plot);
X = [X1(:) X2(:)];
Xval = X + 0.3;
y = double(sum(exp(X),2) > 3);
yval = double(sum(exp(Xval),2) > 3);
[C sigma] = dataset3Params(X, y, Xval, yval)
