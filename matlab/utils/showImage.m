%
% showImage(imageMatrix,referenceIntensity,X,Y,ax)
%
% imageMatrix: input intensity input between 0 and 1.
% referenceIntensity: Optional argument, the imageMatrix is scaled so that
% its mean intensity(clipped) is equal to referenceIntensity. If
% referenceIntensity is 0, no scaling is applied, if it is -1, the image
%   is normalized so that the maximum value==1'. No scaling, nor checking
%   is done when referenceIntensity is left empty ([]), this is the fastest method.
% X,Y: Optional arguments, the pixel positions for display on the axis, otherwise the unit is one pixel
%      If X and Y are scalars, these parameters are used as the top left pixel indexes into an existing image or display screen.
%      If left empty, the default positions are consecutive integers counting from 1 upwards.
% ax: optional argument, axis where to render image. By default a new
% figure window is created. If ax is not an axis but an integer than this
% will be interpreted as the device number on which to display the image in
% full-screen. The first device is numbered 1. Use ALT-TAB followed by "closescreen()" to exit.
%
% Note: the image must have a dimension of at least 2 pixels in X.
%
% Returns a handle to the axis
%
% Example:
%     showImage(getTestImage('boats'),[],[],[],gca); % in window
%     showImage(getTestImage('boats'),[],[],[],2); % in full-screen 2
%     DefaultScreenManager.instance().delete(); % Close all full-screens
%
function res = showImage(imageMatrix,referenceIntensity, X, Y, ax)
    % Do some input checking, and salvage the situation if possible:
    if (islogical(imageMatrix))
        imageMatrix=single(imageMatrix);
    end
    if (~isnumeric(imageMatrix))
        logMessage('Image matrix is not numeric');
        return;
    elseif (isempty(imageMatrix))
        logMessage('Image matrix is empty!');
        return;
    elseif (~any(size(imageMatrix,3)==[1 3 4]))
        message=sprintf('The third dimension of the input imageMatrix should be either 1, 3, or 4, not %d.',size(imageMatrix,3));
        logMessage(message);
        error(message);
    else
        %Convert to floating point
        if ~isfloat(imageMatrix)
            imageMatrix=double(imageMatrix)*2^-16;
        else
            if ndims(imageMatrix)>2 || max(abs(imag(imageMatrix(:)))) < 10*eps(1)
                %Only use the real part
                imageMatrix=real(imageMatrix);
            else
                if (~isreal(imageMatrix))
                    % Convert complex image to false color
                    hue = (angle(imageMatrix)+pi)/(2*pi);
                    hue(hue>=1) = 0;
                    imageMatrix = hsv2rgb(hue,ones(size(imageMatrix)), abs(imageMatrix));
                    clear hue;
                end
            end
        end
        drawingInMatlabAxes=nargin<5 || (ishandle(ax) && strcmpi(get(ax,'Type'),'axes'));
        if (drawingInMatlabAxes)
            %Bring value in range [0,1]
            imageMatrix(imageMatrix<0)=0;
        end
        if (nargin<2 || ~isempty(referenceIntensity))
            if (nargin>1 && ~isempty(referenceIntensity))
                if (referenceIntensity>0)
                    imageMatrix = scaleToIntensity(imageMatrix,referenceIntensity);
                elseif (referenceIntensity<0)
                    maxValue=max(imageMatrix(~isinf(imageMatrix(:))&~isnan(imageMatrix(:))));
                    if isempty(maxValue)
                        error('showImage: The data values are not finite numbers and cannot be displayed.');
                    end
                    imageMatrix=imageMatrix*(-referenceIntensity/maxValue);
                end
            end
        end
        %If drawing in a regular axis, limit the image matrix to [0 1]
        if (drawingInMatlabAxes)
            if ndims(imageMatrix)>2
                maxIntensity=max(imageMatrix,[],3);
                if any(maxIntensity(:)>1.0)
                    maxIntensity=max(1.0,maxIntensity);
                    imageMatrix=imageMatrix.*repmat((1+0)./maxIntensity,[1 1 size(imageMatrix,3)]);
                    imageMatrix(imageMatrix(:)>1)=1; % crop anything that is still too high
                end
                clear maxIntensity;
            else
                imageMatrix(imageMatrix(:)>1)=1;
            end
        end
    end
    % Keep the input image size
    inputImageSize=size(imageMatrix);
    %Check if the offsets are specified
    if (nargin>=4 && ~isempty(X) && ~isempty(Y))
        if ((all(size(X) == inputImageSize(1:2)) || numel(X)==inputImageSize(2)) &&...
            (all(size(Y) == inputImageSize(1:2)) || numel(Y)==inputImageSize(1)))
                rangeSpecified=true;
                offsetSpecified=false;
                %Take a matrix or a vector of axis labels.
                xRange=X(1,:);
                yRange=Y(:,1).';
                if (size(xRange,2)==1)
                    xRange=X(:).'; %Ignore vector direction
                end
                if (size(yRange,2)==1)
                    yRange=Y(:).'; %Ignore vector direction
                end
                clear X Y; % Not needed anymore, may save much memory if these are matrices
        else
            rangeSpecified=false;
            if ((isscalar(X) && isscalar(Y)))
                offsetSpecified=true;
                xOffset=X;
                yOffset=Y;
            else
                error('Size of X and Y should match the dimensions of the input image matrix.');
            end
        end
    else
        rangeSpecified=false;
        offsetSpecified=false;
    end
    %Display in current axes if none is specified. Open new window if required.
    if (nargin<5 || isempty(ax))
        ax=gca();
    end
    %
    %Check if the displayNumber could be a screen device positive integer.
    %
    if (ax>0 && abs(round(ax)-ax)<eps(ax)),
        %Fullscreen
        fullScreenManager=DefaultScreenManager.instance();
        
        displayNumber=ax;

        fullScreenManager.display(displayNumber,imageMatrix,[yOffset xOffset size(imageMatrix,1) size(imageMatrix,2)]);
        %
        % All done drawing in full-screen window
        %
    else
        %
        % Displaying in a regular Matlab figure window
        %
        
        % Copy axes settings for later use
        oldTag=get(ax,'Tag');
        tickDir=get(ax,'TickDir');
        xAxisLocation=get(ax,'XAxisLocation');
        yAxisLocation=get(ax,'YAxisLocation');
        if (length(get(ax,'Children'))==1 && strcmpi(get(get(ax,'Children'),'Type'),'image'))
            %Recycle image object
            im=get(ax,'Children');
        else
            im=image(0,'Parent',ax); %Create a new one
        end
        %Convert grayscale to true color
        if (ndims(imageMatrix)<3 || size(imageMatrix,3)==1),
            imageMatrix=repmat(imageMatrix,[1 1 3]);%Slow, replace with colormap(gray(256)); ?
        end
        try
            if (offsetSpecified)
                % Retrieve the present image
                cData=get(im,'CData');
                oldImageSize=size(cData);
                if (length(oldImageSize)<3)
                    oldImageSize(3)=1;
                end
                % Determine the size in pixels of the final image
                newImageSize=max(oldImageSize,size(imageMatrix)+[yOffset xOffset 0]);
                % Zero-extend the CData array if required
                if (any(oldImageSize<newImageSize))
                    cData(newImageSize(1),newImageSize(2),newImageSize(3))=0;
                end
                % Update the image data
                cData(yOffset+[1:size(imageMatrix,1)],xOffset+[1:size(imageMatrix,2)],:)=imageMatrix;
            else
                % No offset specified, replace the original image
                newImageSize=size(imageMatrix);
                cData=imageMatrix;
            end            
            % Calculate the viewport limits
            if (rangeSpecified)
                if (length(xRange)~=1)
                    halfDiffX=[diff(xRange(1:2))/2 diff(xRange(end-1:end))/2];
                else
                    halfDiffX=[0.5 0.5];
                end
                xLim=[xRange(1)-halfDiffX(1) xRange(end)+halfDiffX(2)];
                if (length(yRange)~=1)
                    halfDiffY=[diff(yRange(1:2))/2 diff(yRange(end-1:end))/2];
                else
                    halfDiffY=[0.5 0.5];
                end
                yLim=[yRange(1)-halfDiffY(1) yRange(end)+halfDiffY(2)];
            else
                % Unity spaced pixels
                if (offsetSpecified)
                    % Show the full image
                    xLim=[0.5 newImageSize(2)+0.5];
                    yLim=[0.5 newImageSize(1)+0.5];
                else
                    % Show only the new bit
                    xLim=[0.5 inputImageSize(2)+0.5];
                    yLim=[0.5 inputImageSize(1)+0.5];
                end
            end
            % set the x-limits and direction
            if (diff(xLim)<0)
                set(ax,'XDir','reverse');
                xLim=-xLim;
            else
                set(ax,'XDir','normal');
            end
            set(ax,'XLim',xLim);
            % set the y-limits and direction
            if (diff(yLim)<0)
                set(ax,'YDir','normal');
                yLim=-yLim;
            else
                set(ax,'YDir','reverse');
            end
            set(ax,'YLim',yLim);
            % Copy the image data to the figure window
            set(im,'CData',cData(:,:,1:3));
            if size(cData,3)>3,
                set(im,'AlphaData',cData(:,:,4));
            end
            clear('cData');
            % Update the axis scales
            if (rangeSpecified)
                set(im,'XData',xRange,'YData',yRange);
            else
                set(im,'XData',[1:newImageSize(2)],'YData',[1:newImageSize(1)]);
            end
        catch Exc
            logMessage('Error in showImage.m: Min value trueColorImageMatrix is %f, and max value is %f.',[min(imageMatrix(:)) max(imageMatrix(:))]);
            logMessage('error message: %s',Exc.message);
            drawingInMatlabAxes
            rethrow(Exc);
        end
        %Restore axes settings
        set(ax,'TickDir',tickDir);
        set(ax,'XAxisLocation',xAxisLocation);
        set(ax,'YAxisLocation',yAxisLocation);
        set(ax,'Tag',oldTag);
        axis(ax, 'equal', 'tight');
    end
    
    if (nargout>0)
        res=ax;
    end
end

function adaptedImageMatrix = scaleToIntensity(imageMatrix,referenceIntensity)
    adaptedImageMatrix=imageMatrix*referenceIntensity/mean(imageMatrix(:));
    meanImageMatrixCropped=mean(min(adaptedImageMatrix(:),1));
    recropTrials=5;
    while abs(meanImageMatrixCropped-referenceIntensity)>.001 && recropTrials>0,
       recropTrials=recropTrials-1;
       adaptedImageMatrix=adaptedImageMatrix*referenceIntensity/meanImageMatrixCropped;
       meanImageMatrixCropped=mean(min(adaptedImageMatrix(:),1));
    end
end
