#!/bin/sh
box_number=$( cat "/home/odroid/boxnumber.txt" )
ffmpeg -use_wallclock_as_timestamps 1 -f video4linux2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 -q 0 -vcodec libx264 -preset superfast -crf 23 -bufsize 2M -y -strict -2 -f segment -segment_list out.list -reset_timestamps 1 -segment_time 300 -segment_atclocktime 1 -strftime 1 "/media/usb-drive/videos/b"${box_number}"_%Y-%m-%d_%H-%M-%S.mp4" > /dev/null 2> /dev/null
