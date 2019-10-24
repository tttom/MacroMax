%
% An example on how to use solveMacroscopicMaxwell to calculate the
% electric vector field distribution in an anisotropic material.
%
function exampleAnisotropic()
  close all;
  
  wavelength = 500e-9;  % m
  k0 = 2*pi / wavelength;
  
  data_shape = [256, 256];
  sample_pitch = [1 1] .* wavelength ./ 4;
  boundary_thickness = 4 * wavelength;
  
  %
  % Define an anisotrpic crystal (Calcite)
  %
  [x_range, y_range] = calcRanges(data_shape, sample_pitch);
  ranges = {x_range, y_range}; 
  % Add prism with optical axis at 45 degrees
  rotZ = @(a) [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];
  rotOptAxis = rotZ(-pi/4);
  epsilon = repmat(eye(3), [1, 1, data_shape]);
  epsilon_crystal = rotOptAxis' * diag([1.4897, 1.6660, 1.6660]).^2 * rotOptAxis; % CaCO3 at 500nm
  epsilon(:, :, floor(end*(1-3/4)/2)+[1:floor(end*3/4)], :) = repmat(epsilon_crystal, [1, 1, floor(data_shape(1)*3/4), data_shape(2)]);
  
  %
  % Define the light source (a periodic current distribution)
  %
  pol_source = [0; 1; 1i];  % y-z-circular polarization
  source = pol_source .* ...
    shiftdim((abs(x_range(:) - (x_range(1) + boundary_thickness)) < 1e-6) .* exp(-y_range.^2./(2 * y_range(round(end*5/8)).^2)), -1)...
    .* exp(1i * k0 .* x_range);
  
  % Add absorbing boundary
  dist_in_boundary = max(max(0,-(ranges{1}(:) - (ranges{1}(1)+boundary_thickness))) + max(0,ranges{1}(:) - (ranges{1}(end)-boundary_thickness)),...
                         max(0,-(ranges{2}(:).' - (ranges{2}(1)+boundary_thickness))) + max(0,ranges{2}(:).' - (ranges{2}(end)-boundary_thickness)));
  weight_boundary = dist_in_boundary ./ boundary_thickness;
  for dimIdx = 1:size(epsilon, 1)
    epsilon(dimIdx,dimIdx,:,:) = epsilon(dimIdx,dimIdx,:,:) + shiftdim(0.25i * (weight_boundary.^1), -2); % define boundary
  end
  % Default the other constitutive relations:
  xi = 0.0;
  zeta = 0.0;
  mu = 1.0;
  
  %
  % Prepare display for showing updates and the result
  %
  figure;
  axs(1) = subplot(2,2,1);
  showImage(shiftdim(epsilon(1, 1, :, :) - 1, 2).' + shiftdim(source(2, :, :), 1).', -1, ranges{1} * 1e6, ranges{2} * 1e6);
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

