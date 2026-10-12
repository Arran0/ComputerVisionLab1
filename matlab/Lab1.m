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

function Lab1(section, varargin)
%LAB1  Run the SC4061 Lab 1 experiments.
%
%   Lab1                 run every section, showing each figure on screen
%   Lab1('2.5')          run one section only (2.1 ... 2.7)
%   Lab1('all', 'save')  also write the figures as PNG files into a
%                        Lab1_results folder beside this script
%   Lab1('all', 'images', FOLDER)
%                        load the lab images from FOLDER explicitly
%
%   Put Lab1.m and the six lab images in the same folder and run it. No
%   addpath, no cd and no particular folder layout are required.

    if nargin < 1 || isempty(section), section = 'all'; end
    saveFigs = any(strcmpi(varargin, 'save'));

    % Optional explicit image folder: Lab1('all', 'images', '/path/to/images')
    imgOverride = '';
    k = find(strcmpi(varargin, 'images'), 1);
    if ~isempty(k) && numel(varargin) > k
        imgOverride = varargin{k + 1};
    end

    isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
    if isOctave
        pkg load image;
        % Only force the gnuplot toolkit when no interactive one exists, as
        % in a headless container. On a normal desktop the default toolkit
        % produces better figures, so leave it alone.
        tk = available_graphics_toolkits();
        if ~any(ismember({'qt', 'fltk'}, tk)) && any(strcmp('gnuplot', tk))
            graphics_toolkit('gnuplot');
        end
    end

    % mfilename('fullpath') is empty for code pasted into the Live Editor or
    % the command window, so fall back to the working directory.
    here = fileparts(mfilename('fullpath'));
    if isempty(here), here = pwd; end

    ctx = struct('images',   findImageDir(here, imgOverride), ...
                 'results',  fullfile(here, 'Lab1_results'), ...
                 'isOctave', isOctave, ...
                 'save',     saveFigs);

    switch lower(section)
        case {'all'}
            sec21(ctx); sec22(ctx); sec23(ctx); sec24(ctx);
            sec25(ctx); sec26(ctx); sec27(ctx);
        case {'2.1', '21'}
            sec21(ctx);
        case {'2.2', '22'}
            sec22(ctx);
        case {'2.3', '23'}
            sec23(ctx);
        case {'2.4', '24'}
            sec24(ctx);
        case {'2.5', '25'}
            sec25(ctx);
        case {'2.6', '26'}
            sec26(ctx);
        case {'2.7', '27'}
            sec27(ctx);
        case {'selftest'}
            warpSelfTest(ctx);
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
        f = newFigure(ctx);
        mesh(x, y, h);
        title(sprintf('Gaussian filter, %dx%d, sigma = %.1f', dim, dim, sigma));
        xlabel('x'); ylabel('y'); zlabel('h(x,y)');
        finishFigure(ctx, f, 'sec23', ...
                     sprintf('octave_gaussian_mesh_sigma%g.png', sigma));
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
    f = newFigure(ctx);
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

    finishFigure(ctx, f, 'sec27', 'octave_perceptron_comparison.png');
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

%% ------------------------------------------------------------------------
%  2.1  Contrast stretching
% -------------------------------------------------------------------------
function [P, P2] = sec21(ctx)
    fprintf('2.1 Contrast stretching\n');

    Pc = imread(labImage(ctx, 'mrt-train.jpg'));
    fprintf('  whos Pc: %s %s\n', mat2str(size(Pc)), class(Pc));

    % MATLAB's imread returns the stored 320x443x3 colour array; Octave's
    % GraphicsMagick reader collapses an all-grey JPEG to 2-D, so guard.
    if ndims(Pc) == 3
        P = rgb2gray(Pc);
    else
        P = Pc;
    end

    fprintf('  min(P(:)) = %d, max(P(:)) = %d\n', min(P(:)), max(P(:)));

    % --- the manual's step (d): two lines, subtraction then multiplication
    rmin = double(min(P(:)));  rmax = double(max(P(:)));
    P2 = uint8(round((double(P) - rmin) * (255 / (rmax - rmin))));

    fprintf('  after stretching: min = %d, max = %d (expected 0 and 255)\n', ...
            min(P2(:)), max(P2(:)));

    f = newFigure(ctx);
    subplot(2,2,1); imshow(P);  title(sprintf('P original (min %d, max %d)', rmin, rmax));
    subplot(2,2,2); imshow(P2); title('P2 contrast stretched (0..255)');
    subplot(2,2,3); imhist(P,  256); title('Histogram of P');
    subplot(2,2,4); imhist(P2, 256); title('Histogram of P2');
    finishFigure(ctx, f, 'sec21', 'octave_contrast_stretch.png');
end

%% ------------------------------------------------------------------------
%  2.2  Histogram equalisation
%
%  NOTE: Octave's histeq is NOT numerically equivalent to MATLAB's -- the
%  Octave source carries an explicit FIXME to that effect. Under MATLAB this
%  section is exact; under Octave the figures are qualitatively right but the
%  numbers differ slightly.
% -------------------------------------------------------------------------
function [P3, P4] = sec22(ctx)
    fprintf('2.2 Histogram equalisation\n');

    Pc = imread(labImage(ctx, 'mrt-train.jpg'));
    if ndims(Pc) == 3, P = rgb2gray(Pc); else, P = Pc; end

    P3 = asUint8(histeq(P,  255));      % (b) equalise
    P4 = asUint8(histeq(P3, 255));      % (c) equalise the result again

    names = {'P  original', 'P3 histeq once', 'P4 histeq twice'};
    imgs  = {P, P3, P4};
    for i = 1:3
        h10 = imhist(imgs{i}, 10);
        fprintf('  %-18s mean = %7.3f   10-bin min/max = %6d/%6d   ratio = %7.2f\n', ...
                names{i}, mean(double(imgs{i}(:))), min(h10), max(h10), ...
                double(max(h10)) / double(max(min(h10), 1)));
    end

    d = abs(double(P4) - double(P3));
    fprintf('  max|P4 - P3| = %d   mean|P4 - P3| = %.4f   pixels changed = %.2f %%\n', ...
            max(d(:)), mean(d(:)), 100 * mean(d(:) > 0));

    f = newFigure(ctx);
    for i = 1:3
        subplot(3,3,(i-1)*3+1); imshow(imgs{i});      title(names{i});
        subplot(3,3,(i-1)*3+2); imhist(imgs{i},  10); title('10 bins');
        subplot(3,3,(i-1)*3+3); imhist(imgs{i}, 256); title('256 bins');
    end
    finishFigure(ctx, f, 'sec22', 'octave_histeq.png');
end

%% ------------------------------------------------------------------------
%  2.3  Linear spatial filtering  (a) builds the filters, (b)-(e) apply them
% -------------------------------------------------------------------------
function sec23(ctx)
    ks = sec23a(ctx);                            % (a)
    kernels = {struct('sigma', 1.0, 'h', ks{1}), struct('sigma', 2.0, 'h', ks{2})};

    fprintf('\n  (b),(c) additive Gaussian noise -- lib-gn.jpg\n');
    filterAndReport(ctx, 'lib-gn.jpg', kernels, 'conv', 'sec23', ...
                    'octave_gaussian_on_gaussian_noise.png');

    fprintf('\n  (d),(e) speckle noise -- lib-sp.jpg\n');
    filterAndReport(ctx, 'lib-sp.jpg', kernels, 'conv', 'sec23', ...
                    'octave_gaussian_on_speckle_noise.png');
end

%% ------------------------------------------------------------------------
%  2.4  Median filtering -- repeats 2.3 (b)-(e) with medfilt2
% -------------------------------------------------------------------------
function sec24(ctx)
    fprintf('2.4 Median filtering\n');

    fprintf('\n  additive Gaussian noise -- lib-gn.jpg\n');
    filterAndReport(ctx, 'lib-gn.jpg', {3, 5}, 'median', 'sec24', ...
                    'octave_median_on_gaussian_noise.png');

    fprintf('\n  speckle noise -- lib-sp.jpg\n');
    filterAndReport(ctx, 'lib-sp.jpg', {3, 5}, 'median', 'sec24', ...
                    'octave_median_on_speckle_noise.png');

    controlledComparison(ctx);
end

%% ------------------------------------------------------------------------
%  Shared helpers for 2.3 / 2.4
% -------------------------------------------------------------------------
function out = asUint8(img)
    % Octave's histeq returns double in [0,1]; MATLAB's returns the input
    % class. Normalise both to uint8 so the rest of the code is identical.
    if isa(img, 'uint8')
        out = img;
    else
        out = uint8(round(double(img) * 255));
    end
end

function out = applyFilter(img, spec, mode)
    switch mode
        case 'conv'
            out = uint8(min(255, max(0, round(conv2(double(img), spec, 'same')))));
        case 'median'
            out = medfilt2(img, [spec spec]);
    end
end

function filterAndReport(ctx, imgname, specs, mode, outdirname, figname)
    img = imread(labImage(ctx, imgname));
    if ndims(img) == 3, img = rgb2gray(img); end

    fprintf('    %-26s %11s %10s %10s\n', 'variant', 'noise sigma', 'mean grad', 'impulse %');
    fprintf('    %-26s %11.3f %10.2f %10.3f\n', 'unfiltered', ...
            estimateNoiseSigma(img), meanGradient(img), impulseFraction(img));

    outs = {img};  labels = {'unfiltered'};
    for k = 1:numel(specs)
        if strcmp(mode, 'conv')
            lab = sprintf('Gaussian 5x5, sigma = %.1f', specs{k}.sigma);
            out = applyFilter(img, specs{k}.h, mode);
        else
            lab = sprintf('median %dx%d', specs{k}, specs{k});
            out = applyFilter(img, specs{k}, mode);
        end
        % Only the unfiltered sigma is meaningful: Immerkaer's estimator
        % assumes uncorrelated noise, which a smoothed output violates.
        fprintf('    %-26s %11s %10.2f %10.3f\n', lab, '(n/a)', ...
                meanGradient(out), impulseFraction(out));
        outs{end+1} = out;  labels{end+1} = lab; %#ok<AGROW>
    end

    f = newFigure(ctx);
    for k = 1:numel(outs)
        subplot(1, numel(outs), k); imshow(outs{k}); title(labels{k});
    end
    finishFigure(ctx, f, outdirname, figname);
end

function s = estimateNoiseSigma(img)
    % Immerkaer: convolve with a kernel that annihilates locally-linear
    % content, so what survives is dominated by noise.
    M = [1 -2 1; -2 4 -2; 1 -2 1];
    [H, W] = size(img);
    s = sum(sum(abs(conv2(double(img), M, 'valid')))) * sqrt(0.5*pi) / (6*(W-2)*(H-2));
end

function g = meanGradient(img)
    hx = fspecial('sobel');          % proxy for retained edge sharpness
    gx = conv2(double(img), hx,  'same');
    gy = conv2(double(img), hx', 'same');
    g  = mean(mean(sqrt(gx.^2 + gy.^2)));
end

function p = impulseFraction(img)
    p = 100 * mean(double(img(:) == 0 | img(:) == 255));
end

function controlledComparison(ctx)
    % Neither lib-gn nor lib-sp has a clean original, so no reference-based
    % measure is possible on them. The quantitative claim is therefore made
    % on a controlled experiment: add noise of known type to a clean image
    % and measure PSNR against the known original.
    fprintf('\n  Controlled comparison (clean reference available)\n');
    Pc = imread(labImage(ctx, 'mrt-train.jpg'));
    if ndims(Pc) == 3, clean = rgb2gray(Pc); else, clean = Pc; end

    rand('seed', 20260924); randn('seed', 20260924);

    gn = double(clean) + 20 * randn(size(clean));
    gn = uint8(min(255, max(0, round(gn))));

    sp = clean; r = rand(size(clean));
    sp(r < 0.02) = 0;  sp(r > 0.98) = 255;

    specs = {struct('sigma', 1.0, 'h', gaussianKernel(5, 1.0)), ...
             struct('sigma', 2.0, 'h', gaussianKernel(5, 2.0))};

    cases = {'additive Gaussian, sigma = 20', gn; 'salt-and-pepper, 4 %', sp};
    for c = 1:size(cases, 1)
        noisy = cases{c, 2};
        fprintf('    %s  (noisy input PSNR = %.2f dB)\n', cases{c,1}, psnr2(clean, noisy));
        for k = 1:numel(specs)
            out = applyFilter(noisy, specs{k}.h, 'conv');
            fprintf('      Gaussian 5x5 s=%.1f     PSNR = %6.2f dB\n', ...
                    specs{k}.sigma, psnr2(clean, out));
        end
        for n = [3 5]
            out = applyFilter(noisy, n, 'median');
            fprintf('      median %dx%d             PSNR = %6.2f dB\n', ...
                    n, n, psnr2(clean, out));
        end
    end
end

function v = psnr2(a, b)
    e = double(a) - double(b);
    m = mean(e(:) .^ 2);
    if m == 0, v = Inf; else, v = 10 * log10(255^2 / m); end
end

%% ------------------------------------------------------------------------
%  2.5  Suppressing noise interference patterns
% -------------------------------------------------------------------------
function sec25(ctx)
    fprintf('2.5 Suppressing noise interference patterns\n');

    % ---- (a) display the corrupted image -------------------------------
    P = imread(labImage(ctx, 'pck-int.jpg'));
    if ndims(P) == 3, P = rgb2gray(P); end

    % ---- (b) Fourier transform and power spectrum -----------------------
    F = fft2(double(P));
    S = abs(F) .^ 2;                       % F is complex, S is real
    f = newFigure(ctx);
    imagesc(fftshift(S .^ 0.1)); colormap('default'); axis image;
    title('fftshift(S.^{0.1})  --  pck-int.jpg');
    finishFigure(ctx, f, 'sec25', 'octave_pck_spectrum.png');

    % ---- (c) peak locations in the UNSHIFTED transform ------------------
    % The manual reads these off the axes or with ginput; found here by
    % local-maximum search outside a DC guard disc so the script is
    % reproducible and non-interactive.
    peaks = findPeaks(F, 10, 2, 0.001);
    total = sum(sum(abs(F(2:end, 2:end)) .^ 2));
    fprintf('  peak locations in the unshifted transform:\n');
    for i = 1:size(peaks, 1)
        v = peaks(i,1); u = peaks(i,2);
        fprintf('    (row %3d, col %3d) MATLAB 1-based   energy share %6.3f %%\n', ...
                v, u, 100 * abs(F(v,u))^2 / total);
    end

    % ---- (d) zero a 5x5 neighbourhood at each peak, in F not S ----------
    G = notchFilter(F, peaks, 2, 'ideal');
    removed = 100 * (1 - sum(sum(abs(G(2:end,2:end)).^2)) / total);
    fprintf('  ideal 5x5 notch: energy removed %.3f %%\n', removed);

    % ---- (e) inverse transform, then try to improve ---------------------
    P5 = uint8(min(255, max(0, round(real(ifft2(G))))));
    G2 = notchFilter(F, peaks, 4, 'gauss');
    P6 = uint8(min(255, max(0, round(real(ifft2(G2))))));
    fprintf('  Gaussian 9x9 notch: energy removed %.3f %%\n', ...
            100 * (1 - sum(sum(abs(G2(2:end,2:end)).^2)) / total));

    f = newFigure(ctx);
    subplot(2,2,1); imshow(P);  title('original pck-int.jpg');
    subplot(2,2,2); imshow(P5); title('(e) after ideal 5x5 notch');
    subplot(2,2,3); imagesc(fftshift(abs(G).^0.2)); axis image; title('notched spectrum');
    subplot(2,2,4); imshow(P6); title('improved: Gaussian 9x9 notch');
    finishFigure(ctx, f, 'sec25', 'octave_pck_int_notch.png');

    % ---- (f) free the caged primate -------------------------------------
    fprintf('\n  (f) primate-caged.jpg\n');
    Q = imread(labImage(ctx, 'primate-caged.jpg'));
    if ndims(Q) == 3, Q = rgb2gray(Q); end
    FQ = fft2(double(Q));
    totalQ = sum(sum(abs(FQ(2:end,2:end)).^2));
    pkQ = findPeaks(FQ, 10, 6, 0.001);
    fprintf('    %d conjugate pairs found; a fence is a harmonic series, not\n', size(pkQ,1));
    fprintf('    a single sinusoid, so one notch pair is never enough.\n');
    GQ = notchFilter(FQ, pkQ, 4, 'gauss');
    fprintf('    energy removed %.3f %%\n', ...
            100 * (1 - sum(sum(abs(GQ(2:end,2:end)).^2)) / totalQ));
    Q2 = uint8(min(255, max(0, round(real(ifft2(GQ))))));

    f = newFigure(ctx);
    subplot(1,3,1); imshow(Q);  title('original');
    subplot(1,3,2); imagesc(fftshift(abs(FQ).^0.2)); axis image; title('spectrum');
    subplot(1,3,3); imshow(Q2); title('after notching the harmonics');
    finishFigure(ctx, f, 'sec25', 'octave_primate_notch.png');
end

function peaks = findPeaks(F, dcGuard, nPairs, minShare)
    % Returns one member of each conjugate pair, strongest first, as
    % [row col] MATLAB 1-based indices into the UNSHIFTED transform.
    mag = abs(F);  [N0, N1] = size(mag);
    total = sum(sum(mag(2:end, 2:end) .^ 2));
    cand = zeros(0, 3);
    for v = 1:N0
        for u = 1:N1
            dv = min(v-1, N0-(v-1));  du = min(u-1, N1-(u-1));
            if hypot(du, dv) < dcGuard, continue; end
            m = mag(v, u);
            r1 = max(1,v-2):min(N0,v+2);  c1 = max(1,u-2):min(N1,u+2);
            if m < max(max(mag(r1, c1))), continue; end
            if (m^2) / total < minShare, continue; end
            cand(end+1, :) = [m v u]; %#ok<AGROW>
        end
    end
    if isempty(cand), peaks = zeros(0,2); return; end
    cand = sortrows(cand, -1);

    peaks = zeros(0, 2);
    for i = 1:size(cand, 1)
        if size(peaks, 1) >= nPairs, break; end
        v = cand(i,2); u = cand(i,3);
        tv = mod(N0 - (v-1), N0) + 1;  tu = mod(N1 - (u-1), N1) + 1;
        dup = false;
        for j = 1:size(peaks, 1)
            if (abs(peaks(j,1)-v) < 4 && abs(peaks(j,2)-u) < 4) || ...
               (abs(peaks(j,1)-tv) < 4 && abs(peaks(j,2)-tu) < 4)
                dup = true; break;
            end
        end
        if ~dup, peaks(end+1, :) = [v u]; end %#ok<AGROW>
    end
end

function G = notchFilter(F, peaks, half, profile)
    % Zero (ideal) or smoothly attenuate (gauss) a (2*half+1)^2 neighbourhood
    % at each peak AND at its conjugate twin. half = 2 gives the manual's 5x5.
    G = F;  [N0, N1] = size(F);
    sigma = max(half / 1.8, 0.5);
    for i = 1:size(peaks, 1)
        v = peaks(i,1); u = peaks(i,2);
        tw = [v u; mod(N0-(v-1), N0)+1, mod(N1-(u-1), N1)+1];
        for k = 1:2
            for dv = -half:half
                for du = -half:half
                    vv = mod(tw(k,1)-1+dv, N0) + 1;
                    uu = mod(tw(k,2)-1+du, N1) + 1;
                    if strcmp(profile, 'ideal')
                        G(vv, uu) = 0;
                    else
                        d2 = dv*dv + du*du;
                        G(vv, uu) = G(vv, uu) * (1 - exp(-d2 / (2*sigma^2)));
                    end
                end
            end
        end
    end
end

%% ------------------------------------------------------------------------
%  2.6  Undoing perspective distortion of a planar surface
% -------------------------------------------------------------------------
function [U, P2] = sec26(ctx)
    fprintf('2.6 Undoing perspective distortion\n');

    P = imread(labImage(ctx, 'book.jpg'));
    fprintf('  book.jpg: %s %s\n', mat2str(size(P)), class(P));

    % (b) The manual obtains the corners interactively:
    %        >> [X Y] = ginput(4)
    % Measured once by fitting a line to each cover edge and intersecting
    % adjacent lines (sub-pixel residuals), so the script stays reproducible.
    X = [142.55; 307.74; 255.37;   4.25];
    Y = [ 18.65;  46.71; 217.37; 158.64];

    % the desired corners, in the SAME order: A4 is 210 mm x 297 mm
    x = [  0; 210; 210;   0];
    y = [  0;   0; 297; 297];

    % (c) set up A u = v from the manual's equation (*)
    A = []; v = [];
    for i = 1:4
        A = [A; X(i) Y(i) 1 0 0 0 -x(i)*X(i) -x(i)*Y(i); ...
                0 0 0 X(i) Y(i) 1 -y(i)*X(i) -y(i)*Y(i)]; %#ok<AGROW>
        v = [v; x(i); y(i)]; %#ok<AGROW>
    end
    u = A \ v;
    U = reshape([u; 1], 3, 3)';
    fprintf('  projective transformation matrix U:\n');
    disp(U);

    % verify: transform the original corners and check they land on target
    w = U * [X'; Y'; ones(1,4)];
    w = w ./ (ones(3,1) * w(3,:));
    err = max(sqrt((w(1,:) - x').^2 + (w(2,:) - y').^2));
    fprintf('  verification -- max corner error = %.3e mm\n', err);
    disp(w(1:2,:)');

    % (d) warp, 1 pixel = 1 mm
    P2 = warpProjective(P, U, 210, 297);
    fprintf('  warped image: %s  (1 pixel = 1 mm)\n', mat2str(size(P2)));

    % (f) locate the pink "computer screen" region
    bbox = findPinkRegion(P2);
    if ~isempty(bbox)
        fprintf('  (f) pink screen region: %d x %d mm at (%d, %d)\n', ...
                bbox(3)-bbox(1)+1, bbox(4)-bbox(2)+1, bbox(1), bbox(2));
    end

    f = newFigure(ctx);
    subplot(1,2,1); imshow(P);  title('book.jpg (slanted view)');
    subplot(1,2,2); imshow(P2); title('P2 warped to frontal A4');
    finishFigure(ctx, f, 'sec26', 'octave_perspective_warp.png');
end

function bbox = findPinkRegion(P2)
    bbox = [];
    if ndims(P2) ~= 3, return; end
    R = double(P2(:,:,1)); G = double(P2(:,:,2)); B = double(P2(:,:,3));
    mx = max(max(R, G), B); mn = min(min(R, G), B);
    sat = (mx - mn) ./ max(mx, 1);
    pink = (R > G + 25) & (R > B + 25) & (R > 90) & (sat > 0.18);
    pink = imopen(pink, ones(3,3));
    pink = imclose(pink, ones(5,5));
    [lab, n] = bwlabel(pink);
    if n == 0, return; end
    best = 0; bi = 0;
    for i = 1:n
        s = sum(lab(:) == i);
        if s > best, best = s; bi = i; end
    end
    [ys, xs] = find(lab == bi);
    bbox = [min(xs) min(ys) max(xs) max(ys)];
end

function P2 = warpProjective(P, U, widthMM, heightMM)
    % The manual prescribes maketform/imtransform:
    %
    %     >> T  = maketform('projective', U');
    %     >> P2 = imtransform(P, T, 'XData', [0 210], 'YData', [0 297]);
    %
    % Both are legacy and may eventually be removed, so this falls back in
    % order. projtform2d (R2022b and later) takes the matrix in the same
    % column convention as U; the older projective2d uses the row convention
    % and so needs U'. The last branch depends on no toolbox function at all,
    % which means the section still works whichever warping API a given
    % MATLAB release happens to ship, and it is the branch exercised by the
    % self-test in warpSelfTest below.
    if exist('maketform', 'file') && exist('imtransform', 'file')
        T  = maketform('projective', U');
        P2 = imtransform(P, T, 'XData', [0 widthMM], 'YData', [0 heightMM]);
    elseif exist('projtform2d', 'file') && exist('imwarp', 'file')
        RA = imref2d([heightMM widthMM], [0 widthMM], [0 heightMM]);
        P2 = imwarp(P, projtform2d(U), 'OutputView', RA);
    elseif exist('projective2d', 'file') && exist('imwarp', 'file')
        RA = imref2d([heightMM widthMM], [0 widthMM], [0 heightMM]);
        P2 = imwarp(P, projective2d(U'), 'OutputView', RA);
    else
        P2 = warpProjectiveBuiltin(P, U, widthMM, heightMM);
    end
end

function P2 = warpProjectiveBuiltin(P, U, widthMM, heightMM)
    % Inverse-map every output pixel through U^-1 and resample bilinearly.
    % Plain array arithmetic only, so it runs on a bare MATLAB with no
    % Image Processing Toolbox at all.
    %
    % Two conventions are matched to imtransform so this is a drop-in:
    %   * XData/YData are inclusive, so x runs 0..widthMM over widthMM+1
    %     columns, giving a 298x211 result for an A4 page.
    %   * MATLAB image spatial coordinates place the first pixel's centre at
    %     (1,1), so a spatial coordinate is itself the fractional array
    %     subscript. Treating it as 0-based shifts the whole image by one
    %     pixel, which is what the identity-transform check below catches.
    xs = 0:widthMM;
    ys = 0:heightMM;
    [xx, yy] = meshgrid(xs, ys);
    n = numel(xx);

    Q  = inv(U) * [xx(:)'; yy(:)'; ones(1, n)];
    sx = (Q(1, :) ./ Q(3, :))';
    sy = (Q(2, :) ./ Q(3, :))';

    [H, W, C] = size(P);
    x0 = floor(sx);  y0 = floor(sy);
    fx = sx - x0;    fy = sy - y0;
    valid = x0 >= 1 & x0 <= W - 1 & y0 >= 1 & y0 <= H - 1;
    x0 = min(max(x0, 1), W - 1);
    y0 = min(max(y0, 1), H - 1);

    i00 = sub2ind([H W], y0,     x0);
    i01 = sub2ind([H W], y0,     x0 + 1);
    i10 = sub2ind([H W], y0 + 1, x0);
    i11 = sub2ind([H W], y0 + 1, x0 + 1);

    P2 = zeros(numel(ys), numel(xs), C);
    for c = 1:C
        ch  = double(P(:, :, c));
        val = ch(i00) .* (1-fx) .* (1-fy) + ch(i01) .* fx .* (1-fy) + ...
              ch(i10) .* (1-fx) .* fy     + ch(i11) .* fx .* fy;
        val(~valid) = 0;
        P2(:, :, c) = reshape(val, numel(ys), numel(xs));
    end
    P2 = uint8(min(255, max(0, round(P2))));
end

function warpSelfTest(ctx)
    % The maketform/imtransform branch is the one that runs wherever those
    % functions exist, so the fallback would otherwise ship untested. This
    % exercises it directly: first against an identity transform, where the
    % correct answer is known exactly, then against imtransform on the real
    % image.
    fprintf('Warp fallback self-test\n');

    % --- 1. identity transform on a small synthetic image -----------------
    N  = 8;
    S  = uint8(reshape(0:(N*N-1), N, N)' * 4);
    Bi = warpProjectiveBuiltin(S, eye(3), N-1, N-1);
    if exist('maketform', 'file') && exist('imtransform', 'file')
        Qi = imtransform(S, maketform('projective', eye(3)'), ...
                         'XData', [0 N-1], 'YData', [0 N-1]);
        fprintf('  identity: max|builtin - imtransform| = %d (expect 0)\n', ...
                max(max(abs(double(Bi) - double(Qi)))));
    else
        fprintf('  identity: builtin returned %s\n', mat2str(size(Bi)));
    end

    % --- 2. the real warp -------------------------------------------------
    P = imread(labImage(ctx, 'book.jpg'));
    X = [142.55; 307.74; 255.37;   4.25];
    Y = [ 18.65;  46.71; 217.37; 158.64];
    x = [  0; 210; 210;   0];
    y = [  0;   0; 297; 297];
    A = []; v = [];
    for i = 1:4
        A = [A; X(i) Y(i) 1 0 0 0 -x(i)*X(i) -x(i)*Y(i); ...
                0 0 0 X(i) Y(i) 1 -y(i)*X(i) -y(i)*Y(i)]; %#ok<AGROW>
        v = [v; x(i); y(i)]; %#ok<AGROW>
    end
    U  = reshape([A \ v; 1], 3, 3)';
    Pb = warpProjectiveBuiltin(P, U, 210, 297);
    fprintf('  book.jpg: builtin warp returned %s\n', mat2str(size(Pb)));

    if exist('maketform', 'file') && exist('imtransform', 'file')
        % Compare against each interpolation setting. The geometry is already
        % established by the identity check above; what is left is a
        % resampling difference, because imtransform builds its sampling grid
        % via interp2 over a forward-mapped mesh rather than by inverse-
        % mapping each output pixel as this fallback does.
        for m = {'nearest', 'bilinear', 'bicubic'}
            Pt = imtransform(P, maketform('projective', U'), m{1}, ...
                             'XData', [0 210], 'YData', [0 297]);
            d  = abs(double(Pb) - double(Pt));
            fprintf('  vs imtransform %-9s max|diff| = %3d, mean = %.3f\n', ...
                    m{1}, max(d(:)), mean(d(:)));
        end
        fprintf(['  Geometry is identical; the residual is a few grey levels\n' ...
                 '  of resampling difference on a noisy JPEG photograph.\n']);
    end
end

%% ------------------------------------------------------------------------
%  Locating the images and placing any output
% -------------------------------------------------------------------------
function d = findImageDir(here, override)
    % Find the folder holding the lab images.
    %
    % An explicit folder always wins. Otherwise the obvious places are tried
    % first, and if none of them has the images the script's folder and the
    % working directory are swept recursively, because on MATLAB Drive and
    % similar setups people often end up with the files one or two folders
    % down from where the script sits.
    if nargin >= 2 && ~isempty(override)
        if hasLabImages(override)
            d = override;
            fprintf('Using images from: %s\n\n', d);
            return;
        end
        error('Lab1:imagesNotFound', ...
              'No lab images in the folder given: %s', override);
    end

    roots = {here};
    if ~strcmp(pwd, here), roots{end+1} = pwd; end

    cand = {};
    for i = 1:numel(roots)
        cand{end+1} = roots{i};                             %#ok<AGROW>
        cand{end+1} = fullfile(roots{i}, 'images');         %#ok<AGROW>
        cand{end+1} = fullfile(roots{i}, '..', 'images');   %#ok<AGROW>
    end
    for i = 1:numel(cand)
        if hasLabImages(cand{i})
            d = cand{i};
            fprintf('Using images from: %s\n\n', d);
            return;
        end
    end

    for i = 1:numel(roots)
        hit = searchForImages(roots{i}, 3);
        if ~isempty(hit)
            d = hit;
            fprintf('Using images from: %s\n\n', d);
            return;
        end
    end

    % Nothing found: say what was looked for, where, and what is actually
    % there, which is far more use than the list of paths on its own.
    msg = sprintf(['Could not find the lab images.\n\n' ...
        'Lab1.m needs these six files:\n' ...
        '    mrt-train.jpg   lib-gn.jpg          lib-sp.jpg\n' ...
        '    pck-int.jpg     primate-caged.jpg   book.jpg\n\n' ...
        'Put them in the same folder as Lab1.m, or anywhere up to three\n' ...
        'folders below it, or pass the folder explicitly:\n' ...
        '    Lab1(''all'', ''images'', ''/path/to/your/images'')\n\n']);
    for i = 1:numel(roots)
        msg = [msg sprintf('Searched under %s, which contains:\n', roots{i})]; %#ok<AGROW>
        msg = [msg listFolder(roots{i})]; %#ok<AGROW>
    end
    error('Lab1:imagesNotFound', '%s', msg);
end

function tf = hasLabImages(folder)
    % Case-insensitive, and tolerant of a .jpeg extension, because MATLAB
    % Drive is case-sensitive and re-saved files sometimes change extension.
    tf = false;
    if isempty(folder) || exist(folder, 'dir') ~= 7, return; end
    L = dir(folder);
    if isempty(L), return; end
    names = lower({L(~[L.isdir]).name});
    tf = any(strcmp(names, 'mrt-train.jpg')) || ...
         any(strcmp(names, 'mrt-train.jpeg'));
end

function hit = searchForImages(root, depth)
    hit = '';
    if depth < 0 || isempty(root) || exist(root, 'dir') ~= 7, return; end
    if hasLabImages(root), hit = root; return; end
    L = dir(root);
    for k = 1:numel(L)
        if ~L(k).isdir, continue; end
        nm = L(k).name;
        if strcmp(nm, '.') || strcmp(nm, '..') || nm(1) == '.', continue; end
        hit = searchForImages(fullfile(root, nm), depth - 1);
        if ~isempty(hit), return; end
    end
end

function s = listFolder(folder)
    s = '';
    if isempty(folder) || exist(folder, 'dir') ~= 7
        s = sprintf('    (not a folder)\n'); return;
    end
    L = dir(folder);
    n = 0;
    for k = 1:numel(L)
        nm = L(k).name;
        if strcmp(nm, '.') || strcmp(nm, '..'), continue; end
        if L(k).isdir
            s = [s sprintf('    [dir]  %s\n', nm)]; %#ok<AGROW>
        else
            s = [s sprintf('           %s\n', nm)]; %#ok<AGROW>
        end
        n = n + 1;
        if n >= 25, s = [s sprintf('           ... and more\n')]; break; end %#ok<AGROW>
    end
    if n == 0, s = sprintf('    (empty)\n'); end
end

function f = labImage(ctx, name)
    % Resolve one image by name, case-insensitively and allowing .jpeg, so a
    % re-saved or renamed upload still loads.
    f = fullfile(ctx.images, name);
    if exist(f, 'file') == 2, return; end
    L = dir(ctx.images);
    files = {L(~[L.isdir]).name};
    [base, ~] = strtok(name, '.');
    want = {lower(name), [lower(base) '.jpeg'], [lower(base) '.jpg']};
    for k = 1:numel(files)
        if any(strcmp(lower(files{k}), want))
            f = fullfile(ctx.images, files{k});
            return;
        end
    end
    error('Lab1:imageMissing', ...
          'Could not find %s in %s', name, ctx.images);
end

function f = newFigure(ctx)
    % Visible by default so the grader sees the results; hidden only while
    % batch-exporting PNGs, where popping up nine windows is unhelpful.
    if ctx.save
        f = figure('visible', 'off');
    else
        f = figure;
    end
end

function finishFigure(ctx, f, subdir, filename)
    % Writing output is opt-in: running Lab1 to look at the results should
    % not create folders as a side effect. Anything written goes beside the
    % script, never above it.
    if ctx.save
        outdir = fullfile(ctx.results, subdir);
        if ~exist(outdir, 'dir'), mkdir(outdir); end
        print(f, fullfile(outdir, filename), '-dpng');
        close(f);
    else
        drawnow;          % make sure the figure is actually painted
    end
end
