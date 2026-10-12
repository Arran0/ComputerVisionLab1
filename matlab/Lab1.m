%% ========================================================================
%  SC4061 / CE4003 / CZ4003  Computer Vision -- Lab 1
%  Point Processing + Spatial Filtering + Frequency Filtering
%  + Imaging Geometry
%
%  Name        : AMARAVEL BABU ARRAN
%  Matric. No. : U2423193C
%  Course      : SC4061 Computer Vision
%
%  This single file contains the complete source code for the lab, as
%  required by Section 3(b) of the lab manual.
%
%  Runs under MATLAB (Image Processing Toolbox) and under GNU Octave with
%  the `image` package; the compatibility shim below reconciles the two.
% =========================================================================

function Lab1(section)
    if nargin < 1, section = 'all'; end

    isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
    if isOctave
        pkg load image;
        graphics_toolkit('gnuplot');
    end

    here    = fileparts(mfilename('fullpath'));
    IMAGES  = fullfile(here, '..', 'images');
    RESULTS = fullfile(here, '..', 'results');

    ctx = struct('images', IMAGES, 'results', RESULTS, 'isOctave', isOctave);

    switch lower(section)
        case {'all'}
            sec23a(ctx);
            sec27(ctx);
        case {'2.3', '23'}
            sec23a(ctx);
        case {'2.7', '27'}
            sec27(ctx);
        otherwise
            error('Lab1:unknownSection', 'Unknown section "%s".', section);
    end
end

%% ------------------------------------------------------------------------
%  2.3(a)  Gaussian averaging filters
%
%           h(x,y) = 1/(2*pi*sigma^2) * exp( -(x^2 + y^2) / (2*sigma^2) )
%
%  normalised so that sum(h(:)) == 1, viewed as a 3-D graph with mesh.
% -------------------------------------------------------------------------
function kernels = sec23a(ctx)
    fprintf('2.3(a) Gaussian averaging filters\n');

    specs   = {5, 1.0; 5, 2.0};
    kernels = cell(size(specs, 1), 1);

    for k = 1:size(specs, 1)
        dim   = specs{k, 1};
        sigma = specs{k, 2};

        h = gaussianKernel(dim, sigma);
        kernels{k} = h;

        fprintf('  dim=%d sigma=%.1f: sum=%.6f min=%.6f max=%.6f\n', ...
                dim, sigma, sum(h(:)), min(h(:)), max(h(:)));
        disp(h);

        r = (dim - 1) / 2;
        [x, y] = meshgrid(-r:r, -r:r);
        f = figure('visible', 'off');
        mesh(x, y, h);
        title(sprintf('Gaussian filter, %dx%d, sigma = %.1f', dim, dim, sigma));
        xlabel('x'); ylabel('y'); zlabel('h(x,y)');
        outdir = fullfile(ctx.results, 'sec23');
        if ~exist(outdir, 'dir'), mkdir(outdir); end
        print(f, fullfile(outdir, sprintf('octave_gaussian_mesh_sigma%g.png', sigma)), '-dpng');
        close(f);
    end
end

%% ------------------------------------------------------------------------
%  Shared helpers
% -------------------------------------------------------------------------
function h = gaussianKernel(dim, sigma)
    r = (dim - 1) / 2;
    [x, y] = meshgrid(-r:r, -r:r);
    h = exp(-(x.^2 + y.^2) / (2 * sigma^2)) / (2 * pi * sigma^2);
    h = h / sum(h(:));   % normalise so the elements sum to 1.0
end

%% ------------------------------------------------------------------------
%  2.7  Code two perceptrons
%
%  Algorithm 1 -- sequential sample update (lecture slides 31-34):
%        x(k) in c1 and w'x <= 0  ->  w(k+1) = w(k) + alpha*x(k)
%        x(k) in c2 and w'x >= 0  ->  w(k+1) = w(k) - alpha*x(k)
%        otherwise                ->  w(k+1) = w(k)
%
%  Algorithm 2 -- batch delta / LMS rule (lecture slide 35):
%        E(w)   = 1/2 * sum_i (r_i - w'x_i)^2
%        dE/dw  = -sum_i (r_i - w'x_i) x_i
%        w(k+1) = w(k) + alpha * sum_i [r_i - w'(k)x_i] x_i
%
%  Slide example: x(1) = [3 3 1]' in c1, x(2) = [1 1 1]' in c2,
%                 alpha = 1, w(1) = [0 0 0]'.
%  The bias is folded into w by the trailing 1 appended to each x.
% -------------------------------------------------------------------------
function sec27(ctx)
    fprintf('2.7 Two perceptrons\n');

    X   = [3 3 1; 1 1 1];      % one sample per row
    cls = [1 2];               % x(1) in c1, x(2) in c2
    r   = [1; -1];             % target labels for Algorithm 2

    % ---- Algorithm 1 ----------------------------------------------------
    [w1, trace, ep1, nUpd] = perceptronAlg1(X, cls, 1.0);
    fprintf('  Algorithm 1 (alpha = 1): converged at k = %d, %d epochs, %d updates\n', ...
            trace(end).k, ep1, nUpd);
    fprintf('    w = [%g %g %g]  ->  boundary  %g*x1 + %g*x2 + %g = 0\n', w1, w1);
    fprintf('    step trace:\n');
    for i = 1:numel(trace)
        fprintf('      k=%2d  x(%d)  wx=%6.1f  rule %d  ->  w = [%g %g %g]\n', ...
                trace(i).k, trace(i).sample, trace(i).a, trace(i).rule, trace(i).w);
    end

    % ---- Algorithm 2 ----------------------------------------------------
    % Stability bound: the batch update is w <- (I - alpha*R)w + alpha*p with
    % R = sum_i x_i x_i', so it converges only for alpha < 2/lambda_max(R).
    R   = X' * X;
    lam = max(eig(R));
    fprintf('\n  Algorithm 2 stability: lambda_max(R) = %.4f, ', lam);
    fprintf('so convergence needs alpha < 2/lambda_max = %.4f\n', 2/lam);

    [~, ~, okDiv] = perceptronAlg2(X, r, 1.0);
    if okDiv
        fprintf('    alpha = 1.0 (as used by Algorithm 1): converged\n');
    else
        fprintf('    alpha = 1.0 (as used by Algorithm 1): DIVERGES, as predicted\n');
    end

    alpha2 = 0.05;
    [w2, costs, ~] = perceptronAlg2(X, r, alpha2);
    fprintf('    alpha = %g: converged in %d epochs, final cost %.3e\n', ...
            alpha2, numel(costs), costs(end));
    fprintf('    w = [%g %g %g]  ->  boundary  %g*x1 + %g*x2 + %g = 0\n', w2, w2);

    % ---- Comparison -----------------------------------------------------
    d1 = (X * w1) / norm(w1(1:2));
    d2 = (X * w2) / norm(w2(1:2));
    fprintf('\n  Comparison of the two boundaries:\n');
    fprintf('    Algorithm 1:  x1 + x2 = %.2f   distances %+.4f / %+.4f   margin ratio %.3f\n', ...
            -w1(3)/w1(1), d1(1), d1(2), min(abs(d1))/max(abs(d1)));
    fprintf('    Algorithm 2:  x1 + x2 = %.2f   distances %+.4f / %+.4f   margin ratio %.3f\n', ...
            -w2(3)/w2(1), d2(1), d2(2), min(abs(d2))/max(abs(d2)));

    % ---- Figures --------------------------------------------------------
    outdir = fullfile(ctx.results, 'sec27');
    if ~exist(outdir, 'dir'), mkdir(outdir); end

    f = figure('visible', 'off');
    subplot(1, 2, 1);
    plot(3, 3, 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'b'); hold on;
    plot(1, 1, 's', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    xs = linspace(-0.5, 4.5, 100);
    plot(xs, (-w1(3) - w1(1)*xs) / w1(2), 'b-',  'LineWidth', 2);
    plot(xs, (-w2(3) - w2(1)*xs) / w2(2), 'g--', 'LineWidth', 2);
    axis([-0.5 4.5 -0.5 4.5]); axis square; grid on;
    xlabel('x_1'); ylabel('x_2'); title('Decision boundaries');
    legend('x(1) in c1', 'x(2) in c2', 'Algorithm 1', 'Algorithm 2', ...
           'Location', 'northeast');
    hold off;

    subplot(1, 2, 2);
    semilogy(costs, 'g-', 'LineWidth', 2); grid on;
    xlabel('Epoch'); ylabel('E(w)');
    title(sprintf('Algorithm 2: cost vs iterations (alpha = %g)', alpha2));

    print(f, fullfile(outdir, 'octave_perceptron_comparison.png'), '-dpng');
    close(f);
end

function [w, trace, epochs, nUpd] = perceptronAlg1(X, cls, alpha)
    w = zeros(size(X, 2), 1);
    k = 0; nUpd = 0; epochs = 0;
    trace = struct('k', {}, 'sample', {}, 'a', {}, 'rule', {}, 'w', {});
    for epoch = 1:100
        epochs  = epoch;
        changed = false;
        for i = 1:size(X, 1)
            k = k + 1;
            a = X(i, :) * w;
            if cls(i) == 1 && a <= 0
                w = w + alpha * X(i, :)';  rule = 1; changed = true; nUpd = nUpd + 1;
            elseif cls(i) == 2 && a >= 0
                w = w - alpha * X(i, :)';  rule = 2; changed = true; nUpd = nUpd + 1;
            else
                rule = 3;
            end
            trace(end+1) = struct('k', k, 'sample', i, 'a', a, 'rule', rule, 'w', w); %#ok<AGROW>
        end
        if ~changed, break; end
    end
end

function [w, costs, ok] = perceptronAlg2(X, r, alpha)
    w = zeros(size(X, 2), 1);
    costs = []; ok = true;
    for it = 1:5000
        e    = r - X * w;
        cost = 0.5 * sum(e .^ 2);
        costs(end+1) = cost; %#ok<AGROW>
        if ~isfinite(cost) || cost > 1e12
            ok = false; return;                 % diverged
        end
        if cost < 1e-12, break; end
        grad = -(X' * e);                       % accumulate over the whole epoch
        w    = w - alpha * grad;
    end
end
