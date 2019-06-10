[fnames,pname] = uigetfile('*.mat','Open AVI files','MultiSelect','On');
%keyboard;
degrees_per_pixel = [1/100,1/100]; %pixels per degree x,y  %%% this value must be checked each time!
for i=1:size(fnames,2)
    load([pname, fnames{i}])
    %frame shifts spline is the variable that contains the xy positions the
    %the 6 lines below are to put it to the correct format. 
    xy = frameshifts_strips_spline.*repmat(degrees_per_pixel,length(frameshifts_strips_spline),1);
    xy = xy - ones(length(xy),1) * mean(xy);
    xwrapped = reshape(xy(:,1), 16, length(xy)/16); % assumes 16 strips per frame 480 hz sampling 30 hz frame rate
    ywrapped = reshape(xy(:,2), 16, length(xy)/16);
    xwrapped = xwrapped - ones(16,1) * xwrapped(1,:);
    ywrapped = ywrapped - ones(16,1) * ywrapped(1,:);
    xwrapped_smoothed = mediansmooth(xwrapped', 20)'; % transposed to smooth on the 2nd dimension, second argument is the range to smooth. Since we have wrapped it now we have one row per frame so 15 is 1/2 second
    ywrapped_smoothed = mediansmooth(ywrapped', 20)';
    xyfixed = xy - [xwrapped_smoothed(:), ywrapped_smoothed(:)];
    eyepositiondata=xyfixed;
    sampletimedata=timeaxis_secs;
    eyepositiondata(2) = eyepositiondata(3) + 1;
eyepositiondata(1) = eyepositiondata(2) - 1;
eyepositiondata(end-1) = eyepositiondata(end-2) + 1;
eyepositiondata(end) = eyepositiondata(end-1) - 1;
eyepositiondata(2,2) = eyepositiondata(3,2) + 1;
eyepositiondata(1,2) = eyepositiondata(2,2) - 1;
eyepositiondata(end-1,2) = eyepositiondata(end-2,2) + 1;
eyepositiondata(end,2) = eyepositiondata(end-1,2) - 1;
eyevel = [0; sqrt(diff(eyepositiondata(:,1)).^2+diff(eyepositiondata(:,2)).^2)./diff(sampletimedata(:))];
    badvel=find(eyevel>200);
    acc=[0;diff(eyevel)./diff(timeaxis_secs)];
    badacc=find(acc>2000);
    blanktimes=find(diff(timeaxis_secs)>0.0022);
    blinkstarts=find(diff(timeaxis_secs)>0.0355);
    if isempty(blinkstarts)
        blinktimes=[];
        blinknum=0;
    else 
        for k=1:size(blinkstarts)
        blinktimeall{k}=timeaxis_secs(blinkstarts(k)):0.002083333333333:timeaxis_secs(blinkstarts(k)+1);
        end
    blinktimes=transpose(cell2mat(blinktimeall));
    blinknum=size(blinkstarts,1);
    end
    badind=unique([badvel;badacc;blanktimes]);
    badtimes=timeaxis_secs(badind);
    xmotionclean=eyepositiondata(:,1);
    ymotionclean=eyepositiondata(:,2);
    timesecsclean=timeaxis_secs;
    velocityclean=eyevel;
    xmotionclean(badind)=[];
    ymotionclean(badind)=[];
    timesecsclean(badind)=[];
    velocityclean(badind)=[];
    lengthtimes=((timeaxis_secs(blanktimes+1)-timeaxis_secs(blanktimes)));
    lengthsamples=round(lengthtimes./0.002083333333333);
    fulltimeaxis=0.0009765625:0.002083333333333:10;
    ymotion=interp1(timesecsclean,ymotionclean,fulltimeaxis,'linear');
    xmotion=interp1(timesecsclean,xmotionclean,fulltimeaxis,'linear');
    velocity=interp1(timesecsclean,velocityclean,fulltimeaxis,'linear');
%     for j=1:length(blanktimes)
%         poorsamples{j}=blanktimes(j):1:(blanktimes(j)+lengthsamples(j));
%         timesecint=[timesecs(1:blanktimes),blanktimes
%     end
    badsamples=4800-length(timeaxis_secs)-length(badvel)-length(badacc);
    if badsamples<796
        timesecs=fulltimeaxis;
        endvalues=2999+find(isnan(velocity(3000:end)));
        beginvalues=find(isnan(velocity(1:2000)));
        velocityvalue=ones(length(endvalues),1)*(velocity(min(endvalues)-1));
        xvalue=ones(length(endvalues),1)*(xmotion(min(endvalues)-1));
        yvalue=ones(length(endvalues),1)*(ymotion(min(endvalues)-1));
        velocitybegvalue=ones(length(beginvalues),1)*(velocity(max(beginvalues)+1));
        xbegvalue=ones(length(beginvalues),1)*(xmotion(max(beginvalues)+1));
        ybegvalue=ones(length(beginvalues),1)*(ymotion(max(beginvalues)+1));
        xmotion(endvalues)=xvalue;
        ymotion(endvalues)=yvalue;
        velocity(endvalues)=velocityvalue;
        xmotion(beginvalues)=xbegvalue;
        ymotion(beginvalues)=ybegvalue;
        velocity(beginvalues)=velocitybegvalue;
        save(['C:\Users\Lenovo\Documents\MATLAB\Envision_extratracescohort1\',fnames{i}],'xmotion','ymotion','badtimes','timesecs','velocity','blinktimes','blinknum')
    end
end
    