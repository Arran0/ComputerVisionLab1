%% ========================================================================
%  SC4061 / CE4003 / CZ4003  Computer Vision -- Lab 1
%  Point Processing + Spatial Filtering + Frequency Filtering
%  + Imaging Geometry
%
%  Name        : <TO BE FILLED>
%  Matric. No. : <TO BE FILLED>
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
        case {'2.3', '23'}
            sec23a(ctx);
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
