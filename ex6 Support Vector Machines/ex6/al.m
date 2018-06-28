model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);
hold on;
l{i}=num2str(C);
i=i+1;
legend (l);
