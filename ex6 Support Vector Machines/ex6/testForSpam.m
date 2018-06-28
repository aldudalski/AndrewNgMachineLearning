function spam = testForSpam (model, filename)

  % Read and predict
  fprintf('\n\n\n==ORIGINAL START===================================================================\n');
  file_contents = readFile(filename)
  fprintf('\n==ORIGINAL END===================================================================\n\n');
  word_indices  = processEmail(file_contents);
  x             = emailFeatures(word_indices);
  p = svmPredict(model, x);

  fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
  fprintf('(1 indicates spam, 0 indicates not spam)\n\n');
