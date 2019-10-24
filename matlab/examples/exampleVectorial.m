%
% An example on how to use solveMacroscopicMaxwell to calculate the
% electric vector field distribution in an isotropic material.
%
function exampleVectorial()
  close all;
  
  wavelength = 500e-9;  % m
  k0 = 2*pi / wavelength;
  
  data_shape = [256, 256];
  sample_pitch = [1 1] .* wavelength ./ 4;
  boundary_thickness = 4 * wavelength;
  
  %
  % Define some glass objects
  %
  [x_range, y_range] = calcRanges(data_shape, sample_pitch);
  ranges = {x_range, y_range};
  n = 1.0 + 0.5 * ((x_range(:) - 3e-6).^2 + (y_range - 1e-6).^2 < 2.5e-6^2) ...
    + 0.5 * ((x_range(:) + 3e-6).^2 + (y_range + 1e-6).^2 < 2.5e-6^2);
  epsilon = n.^2;  % The first two singleton dimensions mark this as a simple isotropic material
 
  %
  % Define the light source (a periodic current distribution)
  %
  pol_source = [0; 1; 1i];  % y-z-circular polarization
  source = pol_source .* ...
            shiftdim((abs(x_range(:) - (x_range(1) + boundary_thickness)) < 1e-6)...
              .* exp(-y_range.^2./(2 * y_range(round(end*5/8)).^2))...
              .* exp(1i * k0 .* x_range(:)), -1);
  
  % Add absorbing boundary
  dist_in_boundary = max(max(0,-(ranges{1}(:) - (ranges{1}(1)+boundary_thickness))) + max(0,ranges{1}(:) - (ranges{1}(end)-boundary_thickness)),...
                         max(0,-(ranges{2}(:).' - (ranges{2}(1)+boundary_thickness))) + max(0,ranges{2}(:).' - (ranges{2}(end)-boundary_thickness)));
  weight_boundary = dist_in_boundary ./ boundary_thickness;
  epsilon = epsilon + 0.25i .* weight_boundary; % define boundary
  % Default the other constitutive relations:
  xi = 0.0;
  zeta = 0.0;
  mu = 1.0;
  
  %
  % Prepare display for showing updates and the result
  %
  figure;
  axs(1) = subplot(2,2,1);
  showImage((epsilon - 1).' + shiftdim(source(1, :, :), 1).', -1, ranges{1} * 1e6, ranges{2} * 1e6);
  title('\chi, S');
  xlabel('x [\mum]');
  ylabel('y [\mum]');
  axis equal tight;
  for pol_idx = 1:3
    axs(1 + pol_idx) = subplot(2,2,1 + pol_idx);
    showImage(zeros(data_shape).', [], ranges{1} * 1e6, ranges{2} * 1e6);
    title(sprintf('E_%s', char('w'+pol_idx)));
    xlabel('x [\mum]');
    ylabel('y [\mum]');
    axis equal tight;
  end
  linkaxes(axs);
  % This function is called after every iteration
  function display_function(it_idx, rms_error, E)
    if ~isempty(it_idx)
      message = logMessage('Iteration %d: %0.3f%%', [it_idx, rms_error*100]);
      set(gcf, 'Name', message);
    end
    
    % Convert vector to array as necessary
    E = reshape(E, [round(numel(E)/prod(data_shape)), data_shape]);

    for display_pol_idx = 1:3
      showImage(shiftdim(E(display_pol_idx,:,:), 1).', -2, ranges{1} * 1e6, ranges{2} * 1e6, axs(1 + display_pol_idx));
    end
  
    drawnow();
  end
  % This function is called after every iteration
  function cont = progress_function(it_idx, rms_error, E)
    if mod(it_idx-1, 10) == 0
     display_function(it_idx, rms_error, E)
    end
    
    cont = rms_error > 1e-3 && it_idx < 1000;  % Decides when to stop the iteration
  end

  %
  % The actual calculation happens in the following line
  % You can give the algorithm singles instead of doubles to save space.
  %
  % helpwin solveMacroscopicMaxwell for information on usage.
  %
  E = solveMacroscopicMaxwell(ranges, k0, epsilon, xi, zeta, mu, source, @progress_function);
  
%   B = calcB(E, sample_pitch, k0);
%   H = calcH(E, sample_pitch, k0, mu, zeta);
%   S = calcS(E, sample_pitch, k0, mu, zeta); % Poynting vector
  
  % Show the final result
  display_function([], [], E);
  logMessage('All done!');  
end

