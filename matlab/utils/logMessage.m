% message = logMessage(message, values)
%
% Logs a message to the command line and to a file called 'log.txt' in the
% current folder with a timestamp prefix.
% The message is formatted using escape characters when a second argument
% is provided,  containing either the substitions, or an empty list [].
%
% Input arguments:
%    message: a text string, which may be formatted using the sprintf
%             syntax (see help sprintf)
%    values: optional values that can be used by a formatted message.
%
% Output arguments:
%    message: the formatted message
%
% Examples:
%    logMessage('This is a simple message.');
%    2013-01-26 17:08:56.589| This is a simple message
%
%    logMessage('This is a formatted message:\n pi=%f, e=%f!',[pi exp(1)]);
%    2013-01-26 17:14:9.627| This is a formatted message:
%     pi=3.141593, e=2.718282!
%
%    logMessage('This is a formatted message\nwithout substitutions.',[]);
%    2013-01-26 17:14:56.122| This is a formatted message
%    without substitutions.
%
function message = logMessage(message, values)
    timeNow=clock();
    if (nargin<1 || isempty(message))
        message='';
    end
    if (isnumeric(message))
        message=num2str(message);
    end
    if (nargin>1)
        message=sprintf(message,values);
    end
    %Prepare the message
    message = sprintf('%04.0f-%02.0f-%02.0f %02.0f:%02.0f:%06.3f| %s',timeNow(1),timeNow(2),timeNow(3),timeNow(4),timeNow(5),timeNow(6),message);
    disp(message);
    %Write the message to file
    fid = fopen('log.txt','a');
    if (fid>0)
        fprintf(fid,'%s\n',message);
        fclose(fid);
    end
end