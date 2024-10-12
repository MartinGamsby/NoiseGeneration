import cv2
import os
import numpy as np
import glob
import math
import textwrap
import gtts
import tempfile
from playsound import playsound
from moviepy.editor import * 
import replicate
import requests
#from urllib.parse import urlparse

from BrownNoiseGeneration import generate


fps = 30

IS_FRENCH=False
LANGUAGE_PREFIX="FR_" if IS_FRENCH else "EN_"

show_images_raw = False#True
show_images_playback_speed = 2 # x
show_images_waitkey = 1#int(1/fps*1000/show_images_playback_speed)+1


 
script_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)))#, "data")

SILENCE_TIME_MS = 100
SILENCE_TIME_SEC = SILENCE_TIME_MS/1000
FONT_HEIGHT_SCALE = 1.5
MARGIN_ON_1920 = 120#120
FONT_SCALING = 10
RENDER_ON_THE_FLY = True#True # (Kind of ... too late...)

MAX_IMAGES = 50

USE_FINAL_LINE = False
FINAL_LINE="Think about it"

# ---------------------------------------------------------------------------------------------------
def multiple_frames(dir, frames):
    dirs = [1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8]
    if abs(dir) > 5:
        return 5
    elif abs(dir) >= 1:
        return math.floor(dir) if dir > 0 else math.ceil(dir)
    for d in dirs:
        if abs(dir) > d:
            return d if dir > 0 else -d
    return 0.0
    
# ---------------------------------------------------------------------------------------------------
def resize_and_zoom(img, to_size, nb_frames, interpolation=cv2.INTER_CUBIC ):

    height, width, layers = img.shape
    
    # Add 1px border for sub pixel translations
    to_size = list(to_size)
    to_size[0] += 2
    to_size[1] += 2
    
    #biggest_side = max(to_size[0],to_size[1])
    #resize_to = biggest_side*1.2
    factor = 1.05
    resize_size = (to_size[0]*factor, to_size[1]*factor)
    biggest_resize_side = max(resize_size[0],resize_size[1])
    biggest_old_side = max(width, height)
    rescale = biggest_resize_side/biggest_old_side
    new_size = (int(width*rescale), int(height*rescale))
    
    #print(resize_size, biggest_resize_side, biggest_old_side, rescale, new_size)
    
    # resized too big:
    resized = cv2.resize(img, new_size, interpolation=interpolation)
    
    # crop middle random:    
    # Calculate the maximum valid starting point for the crop
    max_y = new_size[1] - to_size[1]
    max_x = new_size[0] - to_size[0]

    cropped_random = []
    # Randomly select the starting point for the crop
    start_y = np.random.randint(0, max_y)
    start_x = np.random.randint(0, max_x)
    
    if start_y < (max_y - start_y):
        y_dir = -(start_y/nb_frames)
    else:
        y_dir = ((max_y - start_y)/nb_frames)
    if start_x < (max_x - start_x):
        x_dir = -(start_x/nb_frames)
    else:
        x_dir = ((max_x - start_x)/nb_frames)
    #print( start_y, to_size[0], start_x, to_size[1] )
    #print( y_dir, x_dir )
    x_dir = multiple_frames(x_dir, nb_frames)
    y_dir = multiple_frames(y_dir, nb_frames)
    #print( y_dir, x_dir )
    
    start_y_float = start_y
    start_x_float = start_x
    for i in range(nb_frames):
        # Perform the crop
        start_y = int(start_y_float)
        start_x = int(start_x_float)
        im = resized[start_y:(start_y + to_size[1]), start_x:(start_x + to_size[0])].copy()
        
        # Move by "less than 1" pixels
        x_offset = start_x_float - start_x
        y_offset = start_y_float - start_y
        M = np.float32([
            [1, 0, -x_offset],
            [0, 1, -y_offset]
        ])
        im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))

        
        # Remove 1px border for sub pixel black frame
        #cv2.imshow("im1", im)
        cols = len(im[0])
        rows =len(im)
        im = im[1:(rows-1), 1:(cols-1)]
        #cv2.imshow("im2", im)
        #cv2.waitKey(1)
        
        cropped_random.append( im )#NOTE: We need to copy to write text afterwards!
        if show_images_raw:
            cv2.imshow("ZoomedImage", im)
            cv2.waitKey(1)
            #print(i, nb_frames)
        #print(cropped_random[-1].shape)
        start_y_float += y_dir
        start_x_float += x_dir
    del resized#Delete image?
    return cropped_random
    #print(biggest_side, resize_to, size, )
    
# --------------------------------------------------------------------------------------------------- 
def fade_image_array(img_array, fade_frames, video_writer, fade_backup, output):
    nb_imgs = len(img_array)
    i = 0
    for img in img_array:
        i += 1
        
        if i > (nb_imgs-fade_frames):
            fade_backup.append(img)
        elif fade_backup:
            
            fadein = i/fade_frames
            dst = cv2.addWeighted( fade_backup[0], 1.0-fadein, img, fadein, 0)#linear $
            if RENDER_ON_THE_FLY:
                video_writer.write(dst)
            else:
                output.append(dst)
            fade_backup.pop(0)
                    
        else:
            if RENDER_ON_THE_FLY:
                video_writer.write(img)
            else:
                output.append(img)
        #del img_array[i-i]#img
        img_array[i-1] = np.zeros((1,1), np.uint8)
        #img_array[i-1] = None#Delete image?
        
# --------------------------------------------------------------------------------------------------- 
def fade_images(img_arrays, fade_frames, video_writer):
    output = []
    
    fade_backup = []
    for img_array in img_arrays:
        fade_image_array(img_array, fade_frames, video_writer, fade_backup, output)
    del fade_backup
    return output
    
# --------------------------------------------------------------------------------------------------- 
def get_optimal_font_scale_width(text, width, thickness, scale_start):
    for scale in reversed(range(0, int(scale_start*FONT_SCALING), 1)):#TODO:  Depends on the size of the image??
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/FONT_SCALING, thickness=thickness)
        new_width = textSize[0][0]
        if (new_width <= width):
            #print(f"scale:{scale/10}, width:{width}, new_width:{new_width}, textSize[0][1]:{textSize[0][1]},  text:{text}")
            return scale/FONT_SCALING, textSize[0][1]
    return 1, 10#TODO: Not 10? This is not really possible here? or maybe it is? (len(lines)+0.5)*margin
    
# --------------------------------------------------------------------------------------------------- 
def get_optimal_font_scale_height(lines, height, thickness, font_scale, line_height):
    for scale in reversed(range(0, int(font_scale*FONT_SCALING), 1)):
        textSize = cv2.getTextSize(lines[0], fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/FONT_SCALING, thickness=thickness)
        new_height = textSize[0][1]
        new_height *= len(lines)*FONT_HEIGHT_SCALE
        if (new_height <= height):
            return scale/FONT_SCALING, int(textSize[0][1]*FONT_HEIGHT_SCALE)
    return font_scale, line_height#TODO: Not 10? This is not really possible here? or maybe it is? (len(lines)+0.5)*margin
    
# --------------------------------------------------------------------------------------------------- 
def get_optimal_lines(text, output_size, margin, thickness):
    
    word_width = 16
    landscape = output_size[0] > output_size[1]
    if landscape:
        word_width = int(word_width*output_size[0]/output_size[1])
    lines = textwrap.wrap(text, word_width, break_long_words=False)
    font_scale = 5
    line_height = 10#TODO: Something else by default?
    
      
    
    for line in lines:
        potential_scale, potential_line_height = get_optimal_font_scale_width(line, output_size[0]-margin*2, thickness, 5)#instead of 5, use something else if we want to adjust for get_optimal_font_scale_height in a loop?
        if potential_scale < font_scale:
            font_scale = potential_scale
            line_height = int(potential_line_height*FONT_HEIGHT_SCALE)#buffer because text height is actually smaller than the place it takes...
            
    font_scale, line_height = get_optimal_font_scale_height(lines, output_size[1]-margin*2, thickness, font_scale, line_height)
    # TODO: Try again if it's too high? Change the lines?
    
    #print(font_scale)
    return lines, font_scale, line_height
    
# --------------------------------------------------------------------------------------------------- 
#@profile
def fade_text(img_array, output_size, text, fade_frames, font):
    margin = int((MARGIN_ON_1920 * output_size[1])/1920)
    thickness = math.ceil(3 * max(output_size[1],output_size[0])/1920)
    lines, font_scale, lines_height = get_optimal_lines(text, output_size, margin, thickness)
    lines_top =(len(lines)+0.5)*margin
    #TODO2 (not here) : don't keep the images in RAM, it's a lot for a big video..
    
    
    for image in img_array:        
        text_top = int((output_size[1] - lines_top)/2.)
        org = (margin, text_top)   
        
        for line in lines:
        
            # get boundary of this text
            textsize = cv2.getTextSize(line, font, font_scale, thickness)[0]
            
            # get coords based on boundary
            textX = int((output_size[0] - textsize[0]) / 2)
            org = (textX,text_top)
            image = cv2.putText(image, line, org, font,
                               font_scale, (0,0,0), thickness*3, cv2.LINE_AA)
            image = cv2.putText(image, line, org, font,
                               font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            text_top += lines_height#margin
        #output.extend(img_array)
        #nb_imgs = len(img_array)
        #i = 0
        #for img in img_array:
        #    i += 1
        #    
        #    if i > (nb_imgs-fade_frames):
        #        #print(i, nb_imgs)
        #        fade_backup.append(img)
        #    elif fade_backup:
        #        
        #        fadein = i/fade_frames
        #        #print(i, len(fade_backup), fadein)
        #        dst = cv2.addWeighted( fade_backup[0], 1.0-fadein, img, fadein, 0)#linear $
        #        output.append(dst)
        #        fade_backup.pop(0)
        #                
        #    else:
        #        output.append(img)
   
   
# --------------------------------------------------------------------------------------------------- 
def get_content(content_file_path):
    title = None
    lines = []
    with open(content_file_path, 'r', encoding='UTF-8') as file:
        while l := file.readline():
            line = l.rstrip()
            if line:
                if not title:
                    title = line
                else:
                    lines.append(line)
    if USE_FINAL_LINE:
        lines.append(FINAL_LINE)
    return title, lines
                    
# --------------------------------------------------------------------------------------------------- 
def combine_audio(vidname, audname, outname): 
    import moviepy.editor as mpe
    my_clip = mpe.VideoFileClip(vidname)
    audio_background = mpe.AudioFileClip(audname)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(outname,fps=fps)
    #os.remove(vidname)
    #os.remove(audname)
 
# --------------------------------------------------------------------------------------------------- 
def generate_audio(args, lines, video_directory):

    mp3_names = []
    mp3_lengths = []
    c = 0
    #with tempfile.TemporaryDirectory() as tmp:
    audio_path = os.path.join(script_directory, video_directory, "audio")
    if not os.path.isdir(audio_path):
        os.mkdir(audio_path)
    for i, line in enumerate(lines):
    
        print("\t", line)
        #if args.fast:
        #    sound_file = os.path.join(script_directory, video_directory, 'a' + str(i) +'fast.mp3')
        #else:
        sound_file = os.path.join(audio_path, 'a' + str(i) +'.mp3')
        
        mp3_names.append(sound_file)
        if not os.path.exists(sound_file):
            print("\t\t", sound_file)
            #print(gtts.lang.tts_langs())
            #{'af': 'Afrikaans', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali', 'bs': 'Bosnian', 'ca': 'Catalan', 'cs': 'Czech', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 'es': 'Spanish', 'et': 'Estonian', 'fi': 'Finnish', 'fr': 'French', 'gu': 'Gujarati', 'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian', 'id': 'Indonesian', 'is': 'Icelandic', 'it': 'Italian', 'iw': 'Hebrew', 'ja': 'Japanese', 'jw': 'Javanese', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'la': 'Latin', 'lv': 'Latvian', 'ml': 'Malayalam', 'mr': 'Marathi', 'ms': 'Malay', 'my': 'Myanmar (Burmese)', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'si': 'Sinhala', 'sk': 'Slovak', 'sq': 'Albanian', 'sr': 'Serbian', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai', 'tl': 'Filipino', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'vi': 'Vietnamese', 'zh-CN': 'Chinese (Simplified)', 'zh-TW': 'Chinese (Mandarin/Taiwan)', 'zh': 'Chinese (Mandarin)'}
            tts = gtts.gTTS(line, lang="en", slow=False)#(not args.fast and line != FINAL_LINE))
        
            tts.save(sound_file)
            c+=1
        
        #https://medium.com/@ruslanmv/how-to-create-a-video-with-artificial-intelligence-from-a-text-1dd3ebd49a95
        from mutagen.mp3 import MP3
        audio = MP3(sound_file)
        duration=audio.info.length
        #print("\t\t\tduration: ", duration)

        
        #from IPython.display import Audio
        #from IPython.display import display
        #wn = Audio(sound_file, autoplay=True)
        #display(wn)##
        #del wn

        
        mp3_lengths.append(duration)
        
        #playsound(sound_file)
        #os.remove(sound_file)
    audio_duration = sum(mp3_lengths)
    args.videoDuration = audio_duration# TODO: BVetter than that ... also each image to be different.
    return mp3_names, mp3_lengths, c
 
# --------------------------------------------------------------------------------------------------- 
def merge_audio(args, export_path, mp3_names):
    from pydub import AudioSegment
    silence = AudioSegment.silent(duration=SILENCE_TIME_MS)#TODO?
    full_audio = AudioSegment.empty()    # this will accumulate the entire mp3 audios
    for n, mp3_file in enumerate(mp3_names):
        mp3_file = mp3_file.replace(chr(92), '/')
        print(n, mp3_file)
        
        # Load the current mp3 into `audio_segment`
        audio_segment = AudioSegment.from_mp3(mp3_file)
        
        # Just accumulate the new `audio_segment` + `silence`
        full_audio += audio_segment + silence
        print('Merging ', n)
        
        #NOTE: Also needs "first download ffmpeg from official website and include it in your path. then do pip install ffmpeg and then you will be able to use it without any problems." (https://www.ffmpeg.org/download.html)
    
    # The loop will exit once all files in the list have been used
    # Then export
    full_audio += silence
    full_audio += silence
    full_audio.export(export_path, format='mp3')    
    
# --------------------------------------------------------------------------------------------------- 
def downloadImage(image_url, image_path):
    img_data = requests.get(image_url).content
    with open(image_path, 'wb') as handler:
        handler.write(img_data)
        
# ---------------------------------------------------------------------------------------------------
def generate_image(prompt, video_directory):
    print("There needs to be enough images in ", video_directory)
    sys.exit(1)
    return # Need a valid license
    os.environ["REPLICATE_API_TOKEN"] = ""#TO ADD IN A CONFIG FILE OR WHATEVER
    
    print("running replicate on:", prompt)
    output = replicate.run(
        "stability-ai/sdxl:2b017d9b67edd2ee1401238df49d75da53c523f36e363881e057f5dc3ed3c5b2",
        input={"prompt": prompt},
    )
    print("done running replicate")
    url = output[0]
    print(url, "to")    
    
    output_directory = os.path.join(script_directory, video_directory)
    i = 0    
    while i < MAX_IMAGES:
        output_file = os.path.join(output_directory, "out-" + str(i) + ".png")
        if not os.path.exists(output_file):
            break
        i += 1
    print(output_file)
    downloadImage(url, output_file)
    return output_file

# --------------------------------------------------------------------------------------------------- 
def get_files_from_lines(video_directory, lines, use_last_line):
    files = []
    for filename in glob.glob(video_directory + '\\*.png'):
        files.append(os.path.join(script_directory, filename))
    nb_images = len(files)
    nb_lines = len(lines)
   
    
    
    if nb_lines <= 1 and use_last_line:
        print("Skipping empty content for", video_directory)
        return
    if nb_lines == 0 and not use_last_line:
        print("Skipping empty content for", video_directory)
        return
    
    if nb_images < nb_lines:
        for i in range(nb_images,nb_lines):
            print(i)
            files.append( generate_image(lines[i], video_directory) )
        nb_images = len(files)
   
        if nb_images < nb_lines:
            print(f"oops {nb_images} < {nb_lines}")
            return files
    return files
        
# --------------------------------------------------------------------------------------------------- 
def generate_video(video_directory, override=False):
    
    if not override:
        already_generated = len(glob.glob(video_directory + '\\Final*.mp4')) > 0
        if already_generated:
            print(video_directory, "was already generated")
            return
        
    output_size = (args.width, args.height)
    
    title, lines = get_content(os.path.join(script_directory, video_directory, "content.txt"))
    files = get_files_from_lines(video_directory, lines, use_last_line=USE_FINAL_LINE)

        
    mp3_names, mp3_lengths, c = generate_audio(args, lines, video_directory)
        
    #Merge audio files
    if args.half:
        export_path = os.path.join(video_directory,'audiohalf.mp3')
    elif args.fast:
        export_path = os.path.join(video_directory,'audiofast.mp3')
    else:
        export_path = os.path.join(video_directory,'audio.mp3')
        
    if c > 0 or not os.path.exists(export_path):
        print(mp3_names)
        print("audio duration: ", sum(mp3_lengths))
        merge_audio(args, export_path, mp3_names)
    
        
    #nb_frames_per_image = int(args.videoDuration/nb_images*fps) # including fades
    #nb_frames_fade = int(1.0*fps)# nb_frames_per_image/4
    nb_frames_fade = int((SILENCE_TIME_SEC*2)*fps)
    
    render_everything(video_directory, lines, USE_FINAL_LINE, title, mp3_lengths,
        nb_frames_fade, mp3_name)

    return output
    
# --------------------------------------------------------------------------------------------------- 
def get_image_arrays_from_files(files, lengths, nb_frames_fade, array_cb=None, args2=None):
    img_arrays = []
    output_size = (args.width, args.height)
    print("\n")
    for i, full_path in enumerate(files):
        #print(i, full_path)
        if i == len(lengths):#lines):
            break
        img = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if type(img) == np.ndarray:
            #img_array.append(image_resize(img, output_size[0], output_size[1]))
            #img_array.append(resize_image(img, output_size))
            #img_array.append(crop_square(img, output_size))
            
            #print(img.shape)
            height, width, layers = img.shape
            DALLE_WATERMARK_HEIGHT = 16
            img2 = img[0:(height-DALLE_WATERMARK_HEIGHT), 0:width]
            img_array = resize_and_zoom(img2, output_size, int(lengths[i]*fps + SILENCE_TIME_SEC*fps) + nb_frames_fade)
            del img
            del img2
            #print("nb frames:", int(lengths[i]*fps + SILENCE_TIME_SEC*fps) + nb_frames_fade, len(img_array))
            if array_cb != None:
                array_cb(img_array, args2, i)
            else:
                img_arrays.append(img_array)
            #if show_images_raw:
            #    cv2.imshow("Image", img2)
            #    cv2.waitKey(1)
            if not RENDER_ON_THE_FLY:
                print(f"\rDone {i}", end="")
        else:
            print("failed loading %s" % filename)
    print("\n")
    return img_arrays
    
# --------------------------------------------------------------------------------------------------- 
def get_writer(title):
    if args.half:
        video_path = video_directory +'\\NO AUDIO '+LANGUAGE_PREFIX+"_"+str(args.videoDuration)+"s_"+title+'(HALF).mp4'
        movie_final_path = video_directory +'\\HALF '+LANGUAGE_PREFIX+"_"+str(args.videoDuration)+"s_"+title+'.mp4'
    elif args.fast:
        video_path = video_directory +'\\NO AUDIO '+LANGUAGE_PREFIX+"_"+str(args.videoDuration)+"s_"+title+'(FAST).mp4'
        movie_final_path = video_directory +'\\FAST '+LANGUAGE_PREFIX+"_"+str(args.videoDuration)+"s_"+title+'.mp4'
    else:
        video_path = video_directory +'\\NO AUDIO '+LANGUAGE_PREFIX+"_"+str(args.videoDuration)+"s_"+title+'.mp4'
        movie_final_path = video_directory +'\\Final '+LANGUAGE_PREFIX+"_"+str(args.videoDuration)+"s_"+title+'.mp4'
    output_size = (args.width, args.height)
    
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
    return video_writer, video_path, movie_final_path
    
# --------------------------------------------------------------------------------------------------- 
def render_video(lines, img_arrays, nb_frames_fade, video_writer):

    output_size = (args.width, args.height)
    nb_lines = len(lines)
    print("Adding text")
    for i, img_array in enumerate(img_arrays):
        print("\r", int(i/nb_lines*100), "%", end="")
        fade_text(img_array, output_size, lines[i], nb_frames_fade, font=args.font)
    print("\r", 100, "%")
    img_array = fade_images(img_arrays, int(nb_frames_fade), video_writer)
    del img_arrays#Delete image?
    
    if not RENDER_ON_THE_FLY:
        nb_images = len(img_array)
        print("Rendering video ", nb_images)
        for i in range(nb_images):
            img = img_array[i]
            if args.showImages:
                if show_images_waitkey > 1:
                    cv2.imshow(f"Video preview at ~{show_images_playback_speed}x", img)
                else:
                    cv2.imshow(f"Video rendering", img)
                cv2.waitKey(show_images_waitkey)#*fps)#TODO: *fps because we're faking it there
            video_writer.write(img)
            
            # Deletes the image?
            #del img_array[i]#img
            img_array[i] = np.zeros((1,1), np.uint8)
            #img_array[i] = None#Delete image?
            #del img
            if not (i % 5):
                print("\r", int(i/nb_images*100), "%", end="")
            
    print("\r", 100, "%")
    #del img_arrays#TODO: works?needed?
  
  
# ---------------------------------------------------------------------------------------------------
def maybe_render_part_of_video(img_array, args2, i):
    lines, nb_frames_fade, video_writer, fade_backup = args2
  
    output_size = (args.width, args.height)
    nb_lines = len(lines)
    #print("Adding text")
    print("\r", int(i/nb_lines*100), "%", end="")
    fade_text(img_array, output_size, lines[i], nb_frames_fade, font=args.font)
    fade_image_array(img_array, int(nb_frames_fade), video_writer, fade_backup, None)

  
# ---------------------------------------------------------------------------------------------------
def render_everything(video_directory, lines_image_desc, use_last_line, title, durations, nb_frames_fade, mp3_name):

    #print(title, lines_image_desc)
    #sys.exit(1)
    
    files = get_files_from_lines(video_directory, lines_image_desc, use_last_line=use_last_line)
    video_writer, video_path, movie_final_path = get_writer(title)
        
    if RENDER_ON_THE_FLY:
        fade_backup = []
        img_arrays = get_image_arrays_from_files(files, durations, nb_frames_fade, maybe_render_part_of_video, [lines, nb_frames_fade, video_writer, fade_backup]) #TODO: [item * duration for item in steps]?  [duration-SILENCE_TIME_SEC-nb_frames_fade/fps]??
        del fade_backup
        
    else:
        img_arrays = get_image_arrays_from_files(files, durations, nb_frames_fade) #TODO: [item * duration for item in steps]?  [duration-SILENCE_TIME_SEC-nb_frames_fade/fps]??
        print("render_video: ", render_video(lines, img_arrays, nb_frames_fade, video_writer))
        
    video_writer.release()
    del video_writer

    #Combine audio and video
    clip = VideoFileClip(video_path)  
    duration = clip.duration
    combine_audio(video_path, mp3_name, movie_final_path)
    
    
# --------------------------------------------------------------------------------------------------- 
#import urllib
import urllib.request
import os
import re
#from HTMLParser import HTMLParser
from html.parser import HTMLParser

import yaml

VERBOSE=True
FOLDER_NAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scrape")
DATA_FOLDER_NAME = FOLDER_NAME

# --------------------------------------------------------------------------------------------------- 
class NasaLink():
    def __init__(self, date, caption, url, author=""):
        self.date = date
        self.caption = caption
        self.url = url
        self.author = author
        
    def pretty(self):
        if self.author:
            return "%s (%s) '%s' (by %s)" % ( self.date, self.caption, self.url, self.author )
        else:
            return "%s (%s) '%s'" % ( self.date, self.caption, self.url )
        

# --------------------------------------------------------------------------------------------------- 
class QuotesLinksParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.inLink = False
        self.inAuthor = False
        
        self.lasttag = None
        self.lastvalue = None
        self.lastdata = None
        
        self.links = []
        self.authors = []

    def handle_starttag(self, tag, attrs):
        #self.inLink = False
        #print("inLink=false")
        if tag == 'a':
            #print(attrs)
            #sys.exit()
            for name, value in attrs:
                if name == 'href':# and value.startswith("ap"):# and value == 'Vocabulary':
                    self.lastvalue = value
                    #self.inLink = True
                    #print( value )
                    self.lasttag = tag
                    #print(value)
                if name == "title":
                    if value == "view quote":
                        self.inLink = True
                        #print("\t\tVIEW QUOTE ------------------------------------")
                    if value == "view author":
                        self.inAuthor = True
        if tag == 'div' and self.inLink:
            self.inLink = True
                    

    def handle_endtag(self, tag):
        #print("tag",tag)
        if tag == "div":
            #print("inLink=false2")
            self.inlink = False
        if tag == "a":
            self.inAuthor = False

    def handle_data(self, data):
        #if len(data.strip()):
        #    print(data, self.lasttag, self.inLink)
        #if self.lasttag == 'div' and self.inLink:
        #    print("test data", self.lasttag, self.inLink)
        #    print(data)
        if self.lasttag == 'div' and self.inLink and data.strip():
            link = NasaLink( self.lastdata.strip(), data, self.lastvalue )
            self.links.append( link )
            #print("div", data)
        if self.lasttag == 'a' and self.inAuthor and data.strip():
            #self.links.append( link )
            #print("a", data)
            self.links[-1].author = data.strip()
            
            # TODO: No duplicates!!
            self.authors.append(NasaLink( self.lastdata.strip(), data, self.lastvalue ))
        self.lastdata = data
        

# ---------------------------------------------------------------------------------------------------             
def write_file( name, link ):#TODOOOOOdesc, date, urls ):#url, prefix="", suffix="", folder="" ):
    file_name = name
    with open(os.path.join( DATA_FOLDER_NAME, file_name + ".yaml" ), 'w') as f:
        #f.write(yaml.dump({"desc":desc.encode("utf8"), "date":date.encode("utf8"), "urls":[url.encode("utf8") for url in urls]}))
        #TODO:
        f.write(yaml.dump({"desc":desc.encode("utf8"), "date":date.encode("utf8"), "urls":[url.encode("utf8") for url in urls]}))
    #print( "%s (%s)" % (file_name, desc) )
    #urllib.urlretrieve( url, os.path.join( FOLDER_NAME, folder, whole_name ) )

    
# --------------------------------------------------------------------------------------------------- 
def get_file( url, prefix="", suffix="", folder="" ):
    #https://stackoverflow.com/questions/8286352/how-to-save-an-image-locally-using-python-whose-url-address-i-already-know
    image_name = url.rsplit('/',1)[1]
    whole_name = prefix + image_name + suffix
    whole_name = re.sub('[^0-9a-zA-Z\.\-]+', '_', whole_name)#Remove non-alphanumeric values!
    print( "%s (%s)" % (whole_name, image_name) )
    urllib.urlretrieve( url, os.path.join( FOLDER_NAME, folder, whole_name ) )

# --------------------------------------------------------------------------------------------------- 
def get_content2( url ):
    try:
        #ctx = ssl.create_default_context()
        #ctx.check_hostname = False
        #ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(
            url=url, 
            headers={
              'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36',
              'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
              'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
              'Accept-Encoding': 'none',
              'Accept-Language': 'en-US,en;q=0.8',
              'Connection': 'keep-alive',
              'refere': 'https://example.com',
              'cookie': """your cookie value ( you can get that from your web page) """
            }
        )
        with urllib.request.urlopen(req) as url:
            data = url.read()
            #response = urllib.urlopen(url)
            #data = response.read()      # a `bytes` object
            #text = data.decode('cp1252')#ascii' #utf-8') # a `str`; this step can't be used if data is binary
            text = data.decode('utf-8')
            #TODO: Try all encodings!!!
            return text
    except Exception as e:
        return str(e)        
    return "<NODATA>"



# --------------------------------------------------------------------------------------------------- 
def main(args):
    
    for d in os.listdir("."):
        print(d)
        if os.path.exists(os.path.join(script_directory, d, "content.txt")):
            generate_video(d)
    #sys.exit(1)          
    #          
    #all_categories = [("/topics/age-quotes", "Age"), ("/topics/alone-quotes", "Alone"), ("/topics/amazing-quotes", "Amazing"), ("/topics/anger-quotes", "Anger"), ("/topics/anniversary-quotes", "Anniversary"), ("/topics/architecture-quotes", "Architecture"), ("/topics/art-quotes", "Art"), ("/topics/attitude-quotes", "Attitude"), ("/topics/beauty-quotes", "Beauty"), ("/topics/best-quotes", "Best"), ("/topics/birthday-quotes", "Birthday"), ("/topics/brainy-quotes", "Brainy"), ("/topics/business-quotes", "Business"), ("/topics/car-quotes", "Car"), ("/topics/chance-quotes", "Chance"), ("/topics/change-quotes", "Change"), ("/topics/christmas-quotes", "Christmas"), ("/topics/communication-quotes", "Communication"), ("/topics/computers-quotes", "Computers"), ("/topics/cool-quotes", "Cool"), ("/topics/courage-quotes", "Courage"), ("/topics/dad-quotes", "Dad"), ("/topics/dating-quotes", "Dating"), ("/topics/death-quotes", "Death"), ("/topics/design-quotes", "Design"), ("/topics/diet-quotes", "Diet"), ("/topics/dreams-quotes", "Dreams"), ("/topics/easter-quotes", "Easter"), ("/topics/education-quotes", "Education"), ("/topics/environmental-quotes", "Environmental"), ("/topics/equality-quotes", "Equality"), ("/topics/experience-quotes", "Experience"), ("/topics/experience-quotes", "Experience"), ("/topics/failure-quotes", "Failure"), ("/topics/faith-quotes", "Faith"), ("/topics/family-quotes", "Family"), ("/topics/famous-quotes", "Famous"), ("/topics/fathers-day-quotes", "Fathers Day"), ("/topics/fear-quotes", "Fear"), ("/topics/finance-quotes", "Finance"), ("/topics/fitness-quotes", "Fitness"), ("/topics/food-quotes", "Food"), ("/topics/forgiveness-quotes", "Forgiveness"), ("/topics/freedom-quotes", "Freedom"), ("/topics/friendship-quotes", "Friendship"), ("/topics/funny-quotes", "Funny"), ("/topics/future-quotes", "Future"), ("/topics/gardening-quotes", "Gardening"), ("/topics/god-quotes", "God"), ("/topics/good-quotes", "Good"), ("/topics/government-quotes", "Government"), ("/topics/graduation-quotes", "Graduation"), ("/topics/great-quotes", "Great"), ("/topics/happiness-quotes", "Happiness"), ("/topics/health-quotes", "Health"), ("/topics/history-quotes", "History"), ("/topics/home-quotes", "Home"), ("/topics/hope-quotes", "Hope"), ("/topics/humor-quotes", "Humor"), ("/topics/imagination-quotes", "Imagination"), ("/topics/independence-quotes", "Independence"), ("/topics/inspirational-quotes", "Inspirational"), ("/topics/intelligence-quotes", "Intelligence"), ("/topics/jealousy-quotes", "Jealousy"), ("/topics/jealousy-quotes", "Jealousy"), ("/topics/knowledge-quotes", "Knowledge"), ("/topics/leadership-quotes", "Leadership"), ("/topics/learning-quotes", "Learning"), ("/topics/legal-quotes", "Legal"), ("/topics/life-quotes", "Life"), ("/topics/love-quotes", "Love"), ("/topics/marriage-quotes", "Marriage"), ("/topics/medical-quotes", "Medical"), ("/topics/memorial-day-quotes", "Memorial Day"), ("/topics/men-quotes", "Men"), ("/topics/mom-quotes", "Mom"), ("/topics/money-quotes", "Money"), ("/topics/morning-quotes", "Morning"), ("/topics/mothers-day-quotes", "Mothers Day"), ("/topics/motivational-quotes", "Motivational"), ("/topics/movies-quotes", "Movies"), ("/topics/moving-on-quotes", "Moving On"), ("/topics/music-quotes", "Music"), ("/topics/nature-quotes", "Nature"), ("/topics/new-years-quotes", "New Years"), ("/topics/parenting-quotes", "Parenting"), ("/topics/patience-quotes", "Patience"), ("/topics/patriotism-quotes", "Patriotism"), ("/topics/peace-quotes", "Peace"), ("/topics/pet-quotes", "Pet"), ("/topics/poetry-quotes", "Poetry"), ("/topics/politics-quotes", "Politics"), ("/topics/positive-quotes", "Positive"), ("/topics/power-quotes", "Power"), ("/topics/relationship-quotes", "Relationship"), ("/topics/religion-quotes", "Religion"), ("/topics/religion-quotes", "Religion"), ("/topics/respect-quotes", "Respect"), ("/topics/romantic-quotes", "Romantic"), ("/topics/sad-quotes", "Sad"), ("/topics/saint-patricks-day-quotes", "Saint Patricks Day"), ("/topics/science-quotes", "Science"), ("/topics/smile-quotes", "Smile"), ("/topics/society-quotes", "Society"), ("/topics/space-quotes", "Space"), ("/topics/sports-quotes", "Sports"), ("/topics/strength-quotes", "Strength"), ("/topics/success-quotes", "Success"), ("/topics/sympathy-quotes", "Sympathy"), ("/topics/teacher-quotes", "Teacher"), ("/topics/technology-quotes", "Technology"), ("/topics/teen-quotes", "Teen"), ("/topics/thankful-quotes", "Thankful"), ("/topics/thanksgiving-quotes", "Thanksgiving"), ("/topics/time-quotes", "Time"), ("/topics/travel-quotes", "Travel"), ("/topics/trust-quotes", "Trust"), ("/topics/truth-quotes", "Truth"), ("/topics/valentines-day-quotes", "Valentines Day"), ("/topics/veterans-day-quotes", "Veterans Day"), ("/topics/war-quotes", "War"), ("/topics/wedding-quotes", "Wedding"), ("/topics/wisdom-quotes", "Wisdom"), ("/topics/women-quotes", "Women"), ("/topics/work-quotes", "Work")]
    #for c in all_categories:
    #    url = "https://www.brainyquote.com" + c[0]
    #    page_title = c[1]
    #    #print( page_title, ":", url )
    #    
    ## TODO: Use write_file to save the html.
    #url = "https://www.brainyquote.com/topics/wisdom-quotes"
    ##html = get_content2( url )
    #page_title = "Wisdom"#url[:url.index("/")]
    #print(page_title)
    #html = b'\n<!DOCTYPE html>\n<html lang="en">\n<head>\n<title>Wisdom Quotes - BrainyQuote</title>\n<meta name="robots" content="all">\n<meta charset="utf-8" />\n<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover"><meta name="description" content="Explore 1000 Wisdom Quotes by authors including Socrates, Napoleon Bonaparte, and Lao Tzu at BrainyQuote.">\n<meta name="googlebot" content="NOODP">\n<meta property="ver" content="14.0.1:5669107">\n<meta http-equiv="X-UA-Compatible" content="IE=edge">\n<meta name="msapplication-config" content="none">\n<meta name="apple-mobile-web-app-capable" content="yes">\n<meta property="ts" content="1699848373">\n<meta property="og:site_name" content="BrainyQuote">\n<meta property="og:title" content="Wisdom Quotes - BrainyQuote">\n<meta property="og:type" content="article">\n<meta property="og:description" content="Explore 1000 Wisdom Quotes by authors including Socrates, Napoleon Bonaparte, and Lao Tzu at BrainyQuote.">\n<meta property="og:image" content="https://www.brainyquote.com/photos_tr/en/a/archimedes/101761/archimedes1.jpg">\n<meta property="og:image:type" content="image/jpeg">\n<meta property="og:image:width" content="600">\n<meta property="og:image:height" content="315">\n<meta name="twitter:card" content="summary_large_image">\n<meta name="twitter:site" content="@BrainyQuote">\n<meta name="twitter:app:id:iphone" content="916307096">\n<meta name="twitter:app:id:ipad" content="916307096">\n<meta name="twitter:creator" content="@BrainyQuote">\n<meta name="twitter:title" content="Wisdom Quotes">\n<meta name="twitter:description" content="&quot;Give me a lever long enough and a fulcrum on which to place it, and I shall move the world.&quot; - Archimedes">\n<meta name="twitter:image" content="https://www.brainyquote.com/photos_tr/en/a/archimedes/101761/archimedes1.jpg">\n<meta property="og:url" content="https://www.brainyquote.com/topics/wisdom-quotes">\n<link rel="canonical" href="https://www.brainyquote.com/topics/wisdom-quotes">\n<meta name="apple-itunes-app" content="app-id=916307096">\n<link rel="next" href="https://www.brainyquote.com/topics/wisdom-quotes_2">\n<link rel="icon" href="/favicon.ico?cbv=3" type="image/x-icon">\n<link rel="shortcut icon" href="/favicon.ico?cbv=3" type="image/x-icon">\n<link rel="icon" type="image/png" href="/favicon-32x32.png" sizes="32x32">\n<link rel="icon" type="image/png" href="/favicon-96x96.png" sizes="96x96">\n<link rel="icon" type="image/png" href="/favicon-16x16.png" sizes="16x16">\n<link rel="manifest" href="/manifest.json">\n<meta name="apple-mobile-web-app-title" content="BrainyQuote">\n<meta name="application-name" content="BrainyQuote">\n<meta name="msapplication-TileColor" content="#2b5797">\n<meta name="msapplication-TileImage" content="/mstile-144x144.png">\n<meta name="msapplication-config" content="/browserconfig.xml">\n<meta name="theme-color" content="#ffffff">\n<link href="/st/css/5669107/ws_l_q.css" media="screen, print" type="text/css" rel="stylesheet">\n<script>\nJsLOG = [];\ndocument.BQ_START = new Date().getTime();\nwindow.infoRevn = {adMargin:700,auctTimeoutAdmin:900,bidDivs:false,ctTarg:{"bq_ct":"t","bq_pg":"f","bq_tId":"132601"},ddBullfrog:false,ddTLog:false,delayAuctionMargin:900,firstInfScrollPos:3,forcebdrp:null,gaDisabledBasedOnPageType:false,infScrollSlotsPerAuction:1,infScrollSpacing:6,logPn:false,maxBidEnabled:true,maxBidMult:100,pbSyncDelay:3000,pbSyncsPerBidder:4,revnOn:true,tadpole:false};\nwindow.infoSvr = {RID:"5669107",dReq:true,fsec:true,svrUID:"3a3026f9a035",svrVersion:"14.0.1:5669107"};\nwindow.infoReq = {abgrp:"a",cpageNum:1,curl:"/topics/wisdom-quotes",domainId:"t:132601",gaPageType:"topic",langc:"en",lpageNum:17,vid:"364bc2657c9aca4637fddaa0507870f2"};\nwindow.infoUI = {asyncScripts:["later"],delayedScripts:["kgj2PQz45pb347"],pageUsesGrid:true,pageUsesGridOnly:false,pageUsesListDef:false,redirDom:true};\ndocument.BQ_ADCFG = {"procfg":{"appnexus":{"adj":"0.99"},"indexExchange":{"adj":"0.97"},"openx":{"adj":"0.99","sync":true},"pubmatic":{"adj":"0.95","sync":true},"sovrn":{"adj":"0.99"},"rubicon":{"adj":"0.99"}},"placements":[{"elementId":"div-mobile-320x50","sizes":[[1,1],[320,50]],"nativeAllowed":false,"name":"/1008298/BQ_mobile_320x50","sn":"m-bT","bdrs":{"appnexus":{"uf":true,"bfp":0,"plid":6524274,"adj":0.99},"pubmatic":{"uf":true,"bfp":0,"pubmaticName":"Mobile_Banner"},"rubicon":{"uf":true,"bfp":0,"zone":457876,"sizekey":43,"pos":"atf","adj":0.99},"openx":{"uf":true,"bfp":0,"openxAdSizes":[[320,50]],"oxAdUnitId":538289336},"sovrn":{"uf":true,"bfp":0,"tags":[{"tagName":"Header_BQ_mobile_320x50","tagId":335432,"width":320,"height":50}]},"indexExchange":{"uf":true,"bfp":0,"tags":[{"tagName":"IE_Mobile_inline_11","tagId":175243,"tagNum":11,"width":320,"height":50}]}},"hbSizes":[[320,50]]},{"elementId":"div-gpt-ad-1418667263920-4","sizes":[[300,250],[300,50],[320,50],[320,100],[336,280],[1,1]],"nativeAllowed":false,"name":"/1008298/BQ_ros_square_300x250","sn":"sqT","bdrs":{"appnexus":{"uf":true,"bfp":0,"plid":6524299,"adj":0.99},"pubmatic":{"uf":true,"bfp":0,"pubmaticName":"Square"},"rubicon":{"uf":true,"bfp":0,"zone":457864,"sizekey":15,"pos":"atf","adj":0.99},"openx":{"uf":true,"bfp":0,"oxAdUnitId":538124062},"sovrn":{"uf":true,"bfp":0,"tags":[{"tagName":"Header_BQ_ros_square_300x250","tagId":335425,"width":300,"height":250}]},"indexExchange":{"uf":true,"bfp":0,"tags":[{"tagName":"IE_Desktop_top_2","tagId":175234,"tagNum":2,"width":300,"height":250}]}},"hbSizes":[[300,250],[300,50],[320,50],[320,100],[336,280]]},{"elementId":"div-gpt-ad-1418667263920-3","sizes":[[300,250],[300,50],[320,50],[320,100],[336,280],[1,1]],"nativeAllowed":false,"name":"/1008298/BQ_ros_middle_300x250","sn":"sqM","bdrs":{"appnexus":{"uf":true,"bfp":0,"plid":6524304,"adj":0.99},"pubmatic":{"uf":true,"bfp":0,"pubmaticName":"Middle"},"rubicon":{"uf":true,"bfp":0,"zone":457866,"sizekey":15,"pos":"atf","adj":0.99},"openx":{"uf":true,"bfp":0,"oxAdUnitId":538124063},"sovrn":{"uf":true,"bfp":0,"tags":[{"tagName":"Header_BQ_ros_middle_300x250","tagId":335426,"width":300,"height":250}]},"indexExchange":{"uf":true,"bfp":0,"tags":[{"tagName":"IE_Desktop_mid_5","tagId":175237,"tagNum":5,"width":300,"height":250}]}},"hbSizes":[[300,250],[300,50],[320,50],[320,100],[336,280]]},{"elementId":"div-gpt-ad-1418667263920-2","sizes":[[300,250],[300,50],[320,50],[320,100],[336,280],[1,1]],"nativeAllowed":false,"name":"/1008298/BQ_ros_center_300x250","sn":"sqC","bdrs":{"appnexus":{"uf":true,"bfp":0,"plid":6524307,"adj":0.99},"pubmatic":{"uf":true,"bfp":0,"pubmaticName":"Center"},"rubicon":{"uf":true,"bfp":0,"zone":457868,"sizekey":15,"altsizekey":10,"pos":"atf","adj":0.99},"openx":{"uf":true,"bfp":0,"oxAdUnitId":538124064},"sovrn":{"uf":true,"bfp":0,"tags":[{"tagName":"Header_BQ_ros_center_300x250","tagId":335427,"width":300,"height":250}]},"indexExchange":{"uf":true,"bfp":0,"tags":[{"tagName":"IE_Desktop_center_6","tagId":175238,"tagNum":6,"width":300,"height":250}]}},"hbSizes":[[300,250],[300,50],[320,50],[320,100],[336,280]]},{"elementId":"div-gpt-ad-1418667263920-1","sizes":[[300,250],[300,50],[320,50],[320,100],[336,280],[1,1]],"nativeAllowed":false,"name":"/1008298/BQ_ros_bottom_300x250","sn":"sqB","bdrs":{"appnexus":{"uf":true,"bfp":0,"plid":6524310,"adj":0.99},"pubmatic":{"uf":true,"bfp":0,"pubmaticName":"Bottom"},"rubicon":{"uf":true,"bfp":0,"zone":457870,"sizekey":15,"pos":"atf","adj":0.99},"openx":{"uf":true,"bfp":0,"oxAdUnitId":538124065},"sovrn":{"uf":true,"bfp":0,"tags":[{"tagName":"Header_BQ_ros_bottom_300x250","tagId":335428,"width":300,"height":250}]},"indexExchange":{"uf":true,"bfp":0,"tags":[{"tagName":"IE_Desktop_center_7","tagId":175239,"tagNum":7,"width":300,"height":250}]}},"hbSizes":[[300,250],[300,50],[320,50],[320,100],[336,280]]},{"elementId":"bq_mobile_anchor_placement","sizes":[[1,1],[320,50]],"nativeAllowed":false,"name":"/1008298/BQ_mobile_anchor","sn":"m-bB","bdrs":{"appnexus":{"uf":true,"bfp":0,"plid":6524275,"adj":0.99},"pubmatic":{"uf":true,"bfp":0,"pubmaticName":"Mobile_Anchor"},"rubicon":{"uf":true,"bfp":0,"zone":457878,"sizekey":43,"pos":"atf","adj":0.99},"openx":{"uf":true,"bfp":0,"openxAdSizes":[[320,50]],"oxAdUnitId":538289327},"sovrn":{"uf":true,"bfp":0,"tags":[{"tagName":"Header_BQ_mobile_anchor","tagId":336452,"width":320,"height":50}]},"indexExchange":{"uf":true,"bfp":0,"tags":[{"tagName":"IE_Mobile_anchor_12","tagId":175244,"tagNum":12,"width":320,"height":50}]}},"hbSizes":[[320,50]]}],"curi":"/topics/wisdom-quotes"};\ndocument.BQ_UNIT_LOOKUP = {"/1008298/BQ_ros_bottom_300x250":"div-gpt-ad-1418667263920-1","/1008298/BQ_mobile_320x50":"div-mobile-320x50","/1008298/BQ_mobile_anchor":"bq_mobile_anchor_placement","/1008298/BQ_ros_square_300x250":"div-gpt-ad-1418667263920-4","/1008298/BQ_ros_middle_300x250":"div-gpt-ad-1418667263920-3","/1008298/BQ_ros_center_300x250":"div-gpt-ad-1418667263920-2"};\ndocument.BQ_SLOTS = [["banner","/1008298/BQ_mobile_320x50","div-mobile-320x50"],["main","/1008298/BQ_ros_square_300x250","div-gpt-ad-1418667263920-4"],["mult_auct2","/1008298/BQ_ros_middle_300x250","div-gpt-ad-1418667263920-3"],["mult_auct3","/1008298/BQ_ros_center_300x250","div-gpt-ad-1418667263920-2"],["mult_auct4","/1008298/BQ_ros_bottom_300x250","div-gpt-ad-1418667263920-1"],["mult_auct5","/1008298/BQ_ros_square_300x250","div-gpt-ad-1418667263920-4"],["mult_auct6","/1008298/BQ_ros_middle_300x250","div-gpt-ad-1418667263920-3"],["mult_auct7","/1008298/BQ_ros_center_300x250","div-gpt-ad-1418667263920-2"],["mb_footer","/1008298/BQ_ros_middle_300x250","div-gpt-ad-1418667263920-3"],["mstck","/1008298/BQ_mobile_anchor","bq_mobile_anchor_placement"]];\ndocument.BQ_FOOTER_SLOTS = [];\n</script>\n<script src="/st/js/5669107/bq_first.js" defer></script>\n<script async src="https://fundingchoicesmessages.google.com/i/pub-9038795104372754?ers=1" nonce="qWQkw6lRd-uqlbSgDY9GLw"></script><script nonce="qWQkw6lRd-uqlbSgDY9GLw">(function() {function signalGooglefcPresent() {if (!window.frames[\'googlefcPresent\']) {if (document.body) {const iframe = document.createElement(\'iframe\'); iframe.style = \'width: 0; height: 0; border: none; z-index: -1000; left: -1000px; top: -1000px;\'; iframe.style.display = \'none\'; iframe.name = \'googlefcPresent\'; document.body.appendChild(iframe);} else {setTimeout(signalGooglefcPresent, 0);}}}signalGooglefcPresent();})();</script>\n<script>(function(){\'use strict\';function aa(a){var b=0;return function(){return b<a.length?{done:!1,value:a[b++]}:{done:!0}}}var ba="function"==typeof Object.defineProperties?Object.defineProperty:function(a,b,c){if(a==Array.prototype||a==Object.prototype)return a;a[b]=c.value;return a};\nfunction ea(a){a=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global];for(var b=0;b<a.length;++b){var c=a[b];if(c&&c.Math==Math)return c}throw Error("Cannot find global object");}var fa=ea(this);function ha(a,b){if(b)a:{var c=fa;a=a.split(".");for(var d=0;d<a.length-1;d++){var e=a[d];if(!(e in c))break a;c=c[e]}a=a[a.length-1];d=c[a];b=b(d);b!=d&&null!=b&&ba(c,a,{configurable:!0,writable:!0,value:b})}}\nvar ia="function"==typeof Object.create?Object.create:function(a){function b(){}b.prototype=a;return new b},l;if("function"==typeof Object.setPrototypeOf)l=Object.setPrototypeOf;else{var m;a:{var ja={a:!0},ka={};try{ka.__proto__=ja;m=ka.a;break a}catch(a){}m=!1}l=m?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var la=l;\nfunction n(a,b){a.prototype=ia(b.prototype);a.prototype.constructor=a;if(la)la(a,b);else for(var c in b)if("prototype"!=c)if(Object.defineProperties){var d=Object.getOwnPropertyDescriptor(b,c);d&&Object.defineProperty(a,c,d)}else a[c]=b[c];a.A=b.prototype}function ma(){for(var a=Number(this),b=[],c=a;c<arguments.length;c++)b[c-a]=arguments[c];return b}\nvar na="function"==typeof Object.assign?Object.assign:function(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(d)for(var e in d)Object.prototype.hasOwnProperty.call(d,e)&&(a[e]=d[e])}return a};ha("Object.assign",function(a){return a||na});\nvar p=this||self;function q(a){return a};var t,u;a:{for(var oa=["CLOSURE_FLAGS"],v=p,x=0;x<oa.length;x++)if(v=v[oa[x]],null==v){u=null;break a}u=v}var pa=u&&u[610401301];t=null!=pa?pa:!1;var z,qa=p.navigator;z=qa?qa.userAgentData||null:null;function A(a){return t?z?z.brands.some(function(b){return(b=b.brand)&&-1!=b.indexOf(a)}):!1:!1}function B(a){var b;a:{if(b=p.navigator)if(b=b.userAgent)break a;b=""}return-1!=b.indexOf(a)};function C(){return t?!!z&&0<z.brands.length:!1}function D(){return C()?A("Chromium"):(B("Chrome")||B("CriOS"))&&!(C()?0:B("Edge"))||B("Silk")};var ra=C()?!1:B("Trident")||B("MSIE");!B("Android")||D();D();B("Safari")&&(D()||(C()?0:B("Coast"))||(C()?0:B("Opera"))||(C()?0:B("Edge"))||(C()?A("Microsoft Edge"):B("Edg/"))||C()&&A("Opera"));var sa={},E=null;var ta="undefined"!==typeof Uint8Array,ua=!ra&&"function"===typeof btoa;var F="function"===typeof Symbol&&"symbol"===typeof Symbol()?Symbol():void 0,G=F?function(a,b){a[F]|=b}:function(a,b){void 0!==a.g?a.g|=b:Object.defineProperties(a,{g:{value:b,configurable:!0,writable:!0,enumerable:!1}})};function va(a){var b=H(a);1!==(b&1)&&(Object.isFrozen(a)&&(a=Array.prototype.slice.call(a)),I(a,b|1))}\nvar H=F?function(a){return a[F]|0}:function(a){return a.g|0},J=F?function(a){return a[F]}:function(a){return a.g},I=F?function(a,b){a[F]=b}:function(a,b){void 0!==a.g?a.g=b:Object.defineProperties(a,{g:{value:b,configurable:!0,writable:!0,enumerable:!1}})};function wa(){var a=[];G(a,1);return a}function xa(a,b){I(b,(a|0)&-99)}function K(a,b){I(b,(a|34)&-73)}function L(a){a=a>>11&1023;return 0===a?536870912:a};var M={};function N(a){return null!==a&&"object"===typeof a&&!Array.isArray(a)&&a.constructor===Object}var O,ya=[];I(ya,39);O=Object.freeze(ya);var P;function Q(a,b){P=b;a=new a(b);P=void 0;return a}\nfunction R(a,b,c){null==a&&(a=P);P=void 0;if(null==a){var d=96;c?(a=[c],d|=512):a=[];b&&(d=d&-2095105|(b&1023)<<11)}else{if(!Array.isArray(a))throw Error();d=H(a);if(d&64)return a;d|=64;if(c&&(d|=512,c!==a[0]))throw Error();a:{c=a;var e=c.length;if(e){var f=e-1,g=c[f];if(N(g)){d|=256;b=(d>>9&1)-1;e=f-b;1024<=e&&(za(c,b,g),e=1023);d=d&-2095105|(e&1023)<<11;break a}}b&&(g=(d>>9&1)-1,b=Math.max(b,e-g),1024<b&&(za(c,g,{}),d|=256,b=1023),d=d&-2095105|(b&1023)<<11)}}I(a,d);return a}\nfunction za(a,b,c){for(var d=1023+b,e=a.length,f=d;f<e;f++){var g=a[f];null!=g&&g!==c&&(c[f-b]=g)}a.length=d+1;a[d]=c};function Aa(a){switch(typeof a){case "number":return isFinite(a)?a:String(a);case "boolean":return a?1:0;case "object":if(a&&!Array.isArray(a)&&ta&&null!=a&&a instanceof Uint8Array){if(ua){for(var b="",c=0,d=a.length-10240;c<d;)b+=String.fromCharCode.apply(null,a.subarray(c,c+=10240));b+=String.fromCharCode.apply(null,c?a.subarray(c):a);a=btoa(b)}else{void 0===b&&(b=0);if(!E){E={};c="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split("");d=["+/=","+/","-_=","-_.","-_"];for(var e=\n0;5>e;e++){var f=c.concat(d[e].split(""));sa[e]=f;for(var g=0;g<f.length;g++){var h=f[g];void 0===E[h]&&(E[h]=g)}}}b=sa[b];c=Array(Math.floor(a.length/3));d=b[64]||"";for(e=f=0;f<a.length-2;f+=3){var k=a[f],w=a[f+1];h=a[f+2];g=b[k>>2];k=b[(k&3)<<4|w>>4];w=b[(w&15)<<2|h>>6];h=b[h&63];c[e++]=g+k+w+h}g=0;h=d;switch(a.length-f){case 2:g=a[f+1],h=b[(g&15)<<2]||d;case 1:a=a[f],c[e]=b[a>>2]+b[(a&3)<<4|g>>4]+h+d}a=c.join("")}return a}}return a};function Ba(a,b,c){a=Array.prototype.slice.call(a);var d=a.length,e=b&256?a[d-1]:void 0;d+=e?-1:0;for(b=b&512?1:0;b<d;b++)a[b]=c(a[b]);if(e){b=a[b]={};for(var f in e)Object.prototype.hasOwnProperty.call(e,f)&&(b[f]=c(e[f]))}return a}function Da(a,b,c,d,e,f){if(null!=a){if(Array.isArray(a))a=e&&0==a.length&&H(a)&1?void 0:f&&H(a)&2?a:Ea(a,b,c,void 0!==d,e,f);else if(N(a)){var g={},h;for(h in a)Object.prototype.hasOwnProperty.call(a,h)&&(g[h]=Da(a[h],b,c,d,e,f));a=g}else a=b(a,d);return a}}\nfunction Ea(a,b,c,d,e,f){var g=d||c?H(a):0;d=d?!!(g&32):void 0;a=Array.prototype.slice.call(a);for(var h=0;h<a.length;h++)a[h]=Da(a[h],b,c,d,e,f);c&&c(g,a);return a}function Fa(a){return a.s===M?a.toJSON():Aa(a)};function Ga(a,b,c){c=void 0===c?K:c;if(null!=a){if(ta&&a instanceof Uint8Array)return b?a:new Uint8Array(a);if(Array.isArray(a)){var d=H(a);if(d&2)return a;if(b&&!(d&64)&&(d&32||0===d))return I(a,d|34),a;a=Ea(a,Ga,d&4?K:c,!0,!1,!0);b=H(a);b&4&&b&2&&Object.freeze(a);return a}a.s===M&&(b=a.h,c=J(b),a=c&2?a:Q(a.constructor,Ha(b,c,!0)));return a}}function Ha(a,b,c){var d=c||b&2?K:xa,e=!!(b&32);a=Ba(a,b,function(f){return Ga(f,e,d)});G(a,32|(c?2:0));return a};function Ia(a,b){a=a.h;return Ja(a,J(a),b)}function Ja(a,b,c,d){if(-1===c)return null;if(c>=L(b)){if(b&256)return a[a.length-1][c]}else{var e=a.length;if(d&&b&256&&(d=a[e-1][c],null!=d))return d;b=c+((b>>9&1)-1);if(b<e)return a[b]}}function Ka(a,b,c,d,e){var f=L(b);if(c>=f||e){e=b;if(b&256)f=a[a.length-1];else{if(null==d)return;f=a[f+((b>>9&1)-1)]={};e|=256}f[c]=d;e&=-1025;e!==b&&I(a,e)}else a[c+((b>>9&1)-1)]=d,b&256&&(d=a[a.length-1],c in d&&delete d[c]),b&1024&&I(a,b&-1025)}\nfunction La(a,b){var c=Ma;var d=void 0===d?!1:d;var e=a.h;var f=J(e),g=Ja(e,f,b,d);var h=!1;if(null==g||"object"!==typeof g||(h=Array.isArray(g))||g.s!==M)if(h){var k=h=H(g);0===k&&(k|=f&32);k|=f&2;k!==h&&I(g,k);c=new c(g)}else c=void 0;else c=g;c!==g&&null!=c&&Ka(e,f,b,c,d);e=c;if(null==e)return e;a=a.h;f=J(a);f&2||(g=e,c=g.h,h=J(c),g=h&2?Q(g.constructor,Ha(c,h,!1)):g,g!==e&&(e=g,Ka(a,f,b,e,d)));return e}function Na(a,b){a=Ia(a,b);return null==a||"string"===typeof a?a:void 0}\nfunction Oa(a,b){a=Ia(a,b);return null!=a?a:0}function S(a,b){a=Na(a,b);return null!=a?a:""};function T(a,b,c){this.h=R(a,b,c)}T.prototype.toJSON=function(){var a=Ea(this.h,Fa,void 0,void 0,!1,!1);return Pa(this,a,!0)};T.prototype.s=M;T.prototype.toString=function(){return Pa(this,this.h,!1).toString()};\nfunction Pa(a,b,c){var d=a.constructor.v,e=L(J(c?a.h:b)),f=!1;if(d){if(!c){b=Array.prototype.slice.call(b);var g;if(b.length&&N(g=b[b.length-1]))for(f=0;f<d.length;f++)if(d[f]>=e){Object.assign(b[b.length-1]={},g);break}f=!0}e=b;c=!c;g=J(a.h);a=L(g);g=(g>>9&1)-1;for(var h,k,w=0;w<d.length;w++)if(k=d[w],k<a){k+=g;var r=e[k];null==r?e[k]=c?O:wa():c&&r!==O&&va(r)}else h||(r=void 0,e.length&&N(r=e[e.length-1])?h=r:e.push(h={})),r=h[k],null==h[k]?h[k]=c?O:wa():c&&r!==O&&va(r)}d=b.length;if(!d)return b;\nvar Ca;if(N(h=b[d-1])){a:{var y=h;e={};c=!1;for(var ca in y)Object.prototype.hasOwnProperty.call(y,ca)&&(a=y[ca],Array.isArray(a)&&a!=a&&(c=!0),null!=a?e[ca]=a:c=!0);if(c){for(var rb in e){y=e;break a}y=null}}y!=h&&(Ca=!0);d--}for(;0<d;d--){h=b[d-1];if(null!=h)break;var cb=!0}if(!Ca&&!cb)return b;var da;f?da=b:da=Array.prototype.slice.call(b,0,d);b=da;f&&(b.length=d);y&&b.push(y);return b};function Qa(a){return function(b){if(null==b||""==b)b=new a;else{b=JSON.parse(b);if(!Array.isArray(b))throw Error(void 0);G(b,32);b=Q(a,b)}return b}};function Ra(a){this.h=R(a)}n(Ra,T);var Sa=Qa(Ra);var U;function V(a){this.g=a}V.prototype.toString=function(){return this.g+""};var Ta={};function Ua(){return Math.floor(2147483648*Math.random()).toString(36)+Math.abs(Math.floor(2147483648*Math.random())^Date.now()).toString(36)};function Va(a,b){b=String(b);"application/xhtml+xml"===a.contentType&&(b=b.toLowerCase());return a.createElement(b)}function Wa(a){this.g=a||p.document||document}Wa.prototype.appendChild=function(a,b){a.appendChild(b)};\nfunction Xa(a,b){a.src=b instanceof V&&b.constructor===V?b.g:"type_error:TrustedResourceUrl";var c,d;(c=(b=null==(d=(c=(a.ownerDocument&&a.ownerDocument.defaultView||window).document).querySelector)?void 0:d.call(c,"script[nonce]"))?b.nonce||b.getAttribute("nonce")||"":"")&&a.setAttribute("nonce",c)};function Ya(a){a=void 0===a?document:a;return a.createElement("script")};function Za(a,b,c,d,e,f){try{var g=a.g,h=Ya(g);h.async=!0;Xa(h,b);g.head.appendChild(h);h.addEventListener("load",function(){e();d&&g.head.removeChild(h)});h.addEventListener("error",function(){0<c?Za(a,b,c-1,d,e,f):(d&&g.head.removeChild(h),f())})}catch(k){f()}};var $a=p.atob("aHR0cHM6Ly93d3cuZ3N0YXRpYy5jb20vaW1hZ2VzL2ljb25zL21hdGVyaWFsL3N5c3RlbS8xeC93YXJuaW5nX2FtYmVyXzI0ZHAucG5n"),ab=p.atob("WW91IGFyZSBzZWVpbmcgdGhpcyBtZXNzYWdlIGJlY2F1c2UgYWQgb3Igc2NyaXB0IGJsb2NraW5nIHNvZnR3YXJlIGlzIGludGVyZmVyaW5nIHdpdGggdGhpcyBwYWdlLg=="),bb=p.atob("RGlzYWJsZSBhbnkgYWQgb3Igc2NyaXB0IGJsb2NraW5nIHNvZnR3YXJlLCB0aGVuIHJlbG9hZCB0aGlzIHBhZ2Uu");function db(a,b,c){this.i=a;this.l=new Wa(this.i);this.g=null;this.j=[];this.m=!1;this.u=b;this.o=c}\nfunction eb(a){if(a.i.body&&!a.m){var b=function(){fb(a);p.setTimeout(function(){return gb(a,3)},50)};Za(a.l,a.u,2,!0,function(){p[a.o]||b()},b);a.m=!0}}\nfunction fb(a){for(var b=W(1,5),c=0;c<b;c++){var d=X(a);a.i.body.appendChild(d);a.j.push(d)}b=X(a);b.style.bottom="0";b.style.left="0";b.style.position="fixed";b.style.width=W(100,110).toString()+"%";b.style.zIndex=W(2147483544,2147483644).toString();b.style["background-color"]=hb(249,259,242,252,219,229);b.style["box-shadow"]="0 0 12px #888";b.style.color=hb(0,10,0,10,0,10);b.style.display="flex";b.style["justify-content"]="center";b.style["font-family"]="Roboto, Arial";c=X(a);c.style.width=W(80,\n85).toString()+"%";c.style.maxWidth=W(750,775).toString()+"px";c.style.margin="24px";c.style.display="flex";c.style["align-items"]="flex-start";c.style["justify-content"]="center";d=Va(a.l.g,"IMG");d.className=Ua();d.src=$a;d.alt="Warning icon";d.style.height="24px";d.style.width="24px";d.style["padding-right"]="16px";var e=X(a),f=X(a);f.style["font-weight"]="bold";f.textContent=ab;var g=X(a);g.textContent=bb;Y(a,e,f);Y(a,e,g);Y(a,c,d);Y(a,c,e);Y(a,b,c);a.g=b;a.i.body.appendChild(a.g);b=W(1,5);for(c=\n0;c<b;c++)d=X(a),a.i.body.appendChild(d),a.j.push(d)}function Y(a,b,c){for(var d=W(1,5),e=0;e<d;e++){var f=X(a);b.appendChild(f)}b.appendChild(c);c=W(1,5);for(d=0;d<c;d++)e=X(a),b.appendChild(e)}function W(a,b){return Math.floor(a+Math.random()*(b-a))}function hb(a,b,c,d,e,f){return"rgb("+W(Math.max(a,0),Math.min(b,255)).toString()+","+W(Math.max(c,0),Math.min(d,255)).toString()+","+W(Math.max(e,0),Math.min(f,255)).toString()+")"}function X(a){a=Va(a.l.g,"DIV");a.className=Ua();return a}\nfunction gb(a,b){0>=b||null!=a.g&&0!=a.g.offsetHeight&&0!=a.g.offsetWidth||(ib(a),fb(a),p.setTimeout(function(){return gb(a,b-1)},50))}\nfunction ib(a){var b=a.j;var c="undefined"!=typeof Symbol&&Symbol.iterator&&b[Symbol.iterator];if(c)b=c.call(b);else if("number"==typeof b.length)b={next:aa(b)};else throw Error(String(b)+" is not an iterable or ArrayLike");for(c=b.next();!c.done;c=b.next())(c=c.value)&&c.parentNode&&c.parentNode.removeChild(c);a.j=[];(b=a.g)&&b.parentNode&&b.parentNode.removeChild(b);a.g=null};function jb(a,b,c,d,e){function f(k){document.body?g(document.body):0<k?p.setTimeout(function(){f(k-1)},e):b()}function g(k){k.appendChild(h);p.setTimeout(function(){h?(0!==h.offsetHeight&&0!==h.offsetWidth?b():a(),h.parentNode&&h.parentNode.removeChild(h)):a()},d)}var h=kb(c);f(3)}function kb(a){var b=document.createElement("div");b.className=a;b.style.width="1px";b.style.height="1px";b.style.position="absolute";b.style.left="-10000px";b.style.top="-10000px";b.style.zIndex="-10000";return b};function Ma(a){this.h=R(a)}n(Ma,T);function lb(a){this.h=R(a)}n(lb,T);var mb=Qa(lb);function nb(a){a=Na(a,4)||"";if(void 0===U){var b=null;var c=p.trustedTypes;if(c&&c.createPolicy){try{b=c.createPolicy("goog#html",{createHTML:q,createScript:q,createScriptURL:q})}catch(d){p.console&&p.console.error(d.message)}U=b}else U=b}a=(b=U)?b.createScriptURL(a):a;return new V(a,Ta)};function ob(a,b){this.m=a;this.o=new Wa(a.document);this.g=b;this.j=S(this.g,1);this.u=nb(La(this.g,2));this.i=!1;b=nb(La(this.g,13));this.l=new db(a.document,b,S(this.g,12))}ob.prototype.start=function(){pb(this)};\nfunction pb(a){qb(a);Za(a.o,a.u,3,!1,function(){a:{var b=a.j;var c=p.btoa(b);if(c=p[c]){try{var d=Sa(p.atob(c))}catch(e){b=!1;break a}b=b===Na(d,1)}else b=!1}b?Z(a,S(a.g,14)):(Z(a,S(a.g,8)),eb(a.l))},function(){jb(function(){Z(a,S(a.g,7));eb(a.l)},function(){return Z(a,S(a.g,6))},S(a.g,9),Oa(a.g,10),Oa(a.g,11))})}function Z(a,b){a.i||(a.i=!0,a=new a.m.XMLHttpRequest,a.open("GET",b,!0),a.send())}function qb(a){var b=p.btoa(a.j);a.m[b]&&Z(a,S(a.g,5))};(function(a,b){p[a]=function(){var c=ma.apply(0,arguments);p[a]=function(){};b.apply(null,c)}})("__h82AlnkH6D91__",function(a){"function"===typeof window.atob&&(new ob(window,mb(window.atob(a)))).start()});}).call(this);\nwindow.__h82AlnkH6D91__("WyJwdWItOTAzODc5NTEwNDM3Mjc1NCIsW251bGwsbnVsbCxudWxsLCJodHRwczovL2Z1bmRpbmdjaG9pY2VzbWVzc2FnZXMuZ29vZ2xlLmNvbS9iL3B1Yi05MDM4Nzk1MTA0MzcyNzU0Il0sbnVsbCxudWxsLCJodHRwczovL2Z1bmRpbmdjaG9pY2VzbWVzc2FnZXMuZ29vZ2xlLmNvbS9lbC9BR1NLV3hYeTNiQmpycjREYmxBeFdKbTZYSEtYNFpjVnRXaDBKVWpWR0hpLXhXRjZXY0xlZFpUSXhEODJUclpsTzBGb1k1QzhTQmJGN1IyYlNFNllINnNzY25mMC1nXHUwMDNkXHUwMDNkP3RlXHUwMDNkVE9LRU5fRVhQT1NFRCIsImh0dHBzOi8vZnVuZGluZ2Nob2ljZXNtZXNzYWdlcy5nb29nbGUuY29tL2VsL0FHU0tXeFVCbmgya3FTeEVDX2ZUaVY0d0FRcmtWQ2tnWVVuSTRJaHBLbGhNNGZrMGJuaFBXQi1WSURlVTBBZzlLNVVQVjd0aEpxZUJDSTZoT1QxZnBHT0drUjBGclFcdTAwM2RcdTAwM2Q/YWJcdTAwM2QxXHUwMDI2c2JmXHUwMDNkMSIsImh0dHBzOi8vZnVuZGluZ2Nob2ljZXNtZXNzYWdlcy5nb29nbGUuY29tL2VsL0FHU0tXeFVmY2duaVVPOXV3ak1WRmJRRTA1VC1oUjR5X252WEJrODI4MVFVakJQal9nbzFlc2ItZ0loRUNfM1otTHYtNXBRWmRGSWc1UWRDcmZadmQ4NHU5Y3I5SWdcdTAwM2RcdTAwM2Q/YWJcdTAwM2QyXHUwMDI2c2JmXHUwMDNkMSIsImh0dHBzOi8vZnVuZGluZ2Nob2ljZXNtZXNzYWdlcy5nb29nbGUuY29tL2VsL0FHU0tXeFZiSmI4TEtsVHRVYkxhUmM4ZnBHSThWSmtpQnlfdHhMUUdLR2Q2SENUbmxiS3UtQjlLUVk0TTJMWjU0SzFpUzlaMWh0cEZOTmtFUm1sN0RPM3hSQUVzLXdcdTAwM2RcdTAwM2Q/c2JmXHUwMDNkMiIsImRpdi1ncHQtYWQiLDIwLDEwMCwiY0hWaUxUa3dNemczT1RVeE1EUXpOekkzTlRRXHUwMDNkIixbbnVsbCxudWxsLG51bGwsImh0dHBzOi8vd3d3LmdzdGF0aWMuY29tLzBlbW4vZi9wL3B1Yi05MDM4Nzk1MTA0MzcyNzU0LmpzP3VzcXBcdTAwM2RDQUUiXSwiaHR0cHM6Ly9mdW5kaW5nY2hvaWNlc21lc3NhZ2VzLmdvb2dsZS5jb20vZWwvQUdTS1d4V2ZjY2g0QnNHcHJDR2p5b1g3Q01TcGdkU09tU3haeGJDRWwwZl9LMmphZE8wY3VZcEx1RW1jaDZvZnBrTktPTk9CUTl5OW9hakRIaE5ZVzdwVXI3Z1VFUVx1MDAzZFx1MDAzZCJd");\n</script>\n</head>\n<body class=" qll-new is-grid-md gr-li-h ">\n<nav>\n<div id="bqNavBarCtrl">\n<div id="bq-tn-id" class="bq-tnav navbar">\n<div class="bq-bluebar" style="line-height:10px">\n<div id="bq-tnav-hide" class="bq-tnav-right grid-layout-hide"> <div class="bq-top-b-a-c"> <div class="bq-top-b-a"> <div id="div-mobile-320x50-banner_edge" class=" "> <div id="div-mobile-320x50-banner" class="bq_ad_mobile_banner_cls bqAdCollapse "></div> </div> </div> </div> <a href="/search" class="bq-tnav-btn bq-nav-search-icon" aria-label="Search"><img src="/st/img/5669107/fa/search.svg" class="s-btn bq-fa-invert" alt="Search" aria-hidden="true" /></a> <div class="bq-nav-search-form"> <form action="/search_results" class="no-border bq-no-print form-search" style="padding-left:8px; padding-right:8px" autocomplete="off"> <div class="bq-search" role="search"> <input type="image" src="/st/img/5669107/fa/search.svg" class="s-btn bq-fa" alt="Submit Form" aria-label="Search" /> <input id="bq-search-input" autocapitalize="off" spellcheck="false" aria-label="Search on BrainyQuote" type="text" placeholder="search" maxlength="80" name="q" class="s-fld-t input-medium search-query s-small" value /> </div> </form> </div> <div class="nav-item dropdown bq-hamc" style="right: 0"> <a href="#" class="bq-tnav-btn bars dropdown-toggle" data-bs-toggle="dropdown" aria-expanded="false" aria-label="Menu"> <img src="/st/img/5669107/fa/bars.svg" alt="Menu" class="bq-fa-invert"> </a> <div class="bq-right-menu hamburger-items dropdown-menu dropdown-menu-end"> <div class="bq-ham-ctr"> <div class="row bq-ham-m"> <div class="col-6 bq-rm-col"> <div class="tn-small-header ">Site</div> <div class="dropdown-item bq-ni "> <a href="/">Home</a> </div> <div class="dropdown-item bq-ni "> <a href="/authors">Authors</a> </div> <div class="dropdown-item bq-ni "> <a href="/topics">Topics</a> </div> <div class="dropdown-item bq-ni "> <a href="/quote_of_the_day">Quote Of The Day</a> </div> <div class="dropdown-item bq-ni "> <a href="/top_100_quotes">Top 100 Quotes</a> </div> <div class="dropdown-item bq-ni "> <a href="/profession/">Professions</a> </div> <div class="dropdown-item bq-ni "> <a href="/birthdays/">Birthdays</a> </div> </div> <div class="col-6 bq-rm-col"> <div class="tn-small-header ">About</div> <div class="dropdown-item bq-ni "> <a href="/about/">About Us</a> </div> <div class="dropdown-item bq-ni "> <a href="/about/contact_us">Contact Us</a> </div> <div class="dropdown-item bq-ni "> <a href="/about/terms">Terms Of Service</a> </div> <div class="dropdown-item bq-ni bq_eu"> <a href="#" onclick="event.preventDefault();web_ui_later.TcfUserConsent.openConsentSettings();return false;"> Privacy Settings</a> </div> <div class="dropdown-item bq-ni "> <a href="/about/privacy">Privacy Policy</a> </div> </div> <div class="col-12 bq-right-menu-footer"> <a href="/about/copyright" class="bq-copyr">Copyright</a> &#169; 2001 - 2023 BrainyQuote </div> </div> </div> </div> </div> </div>\n<a href="/" class="brand" style="padding:0px 5px 0px 15px; color:white" aria-label="Home Page"><img src="/st/img/5669107/brainyquote_sl@2x.png" alt="BrainyQuote" class="bqLogoImg"></a>\n<div class="bq-nav-search-form sm-mbl hide-on-landscape">\n<form action="/search_results" class="no-border bq-no-print form-search" style="padding-left:8px; padding-right:8px" autocomplete="off">\n<div class="bq-search" role="search">\n<input type="image" src="/st/img/5669107/fa/search.svg" class="s-btn bq-fa" alt="Submit Form" aria-label="Search" />\n<input id="bq-search-input-mbl-por" autocapitalize="off" spellcheck="false" aria-label="Search on BrainyQuote" type="text" placeholder="search" maxlength="80" name="q" class="s-fld-t input-medium search-query s-small" value />\n</div>\n</form>\n</div>\n<ul class="nav navbar-nav hidden-xs bq-top-nav-r">\n<li class="bq-ni nav-item nav-item"><a href="/" class="txnav nav-link">Home</a></li>\n<li class="bq-ni nav-item nav-item"><a href="/authors" class="txnav nav-link">Authors</a></li>\n<li class="bq-ni nav-item nav-item active"><a href="/topics" class="txnav nav-link">Topics</a></li>\n<li class="bq-ni nav-item nav-item"><a href="/quote_of_the_day" class="txnav nav-link">Quote Of The Day</a></li>\n</ul>\n</div>\n</div>\n<div class="bq-subnav">\n<div class="navbar bq-navbar" style="margin-left:0; margin-right:0">\n<div class="navbar-header">\n<h1 class="bq-subnav-h1">\nWisdom Quotes\n</h1>\n</div>\n<div id="bq-subnav-hide" class="r_dt grid-layout-hide">\n<ul class="nav navbar-right hidden-1col">\n<li class="hidden-1col dropdown">\n\n<a href="#" class="dropdown-toggle" data-bs-toggle="dropdown" role="button" aria-haspopup="true" aria-expanded="false" aria-label="Popular Authors"> Popular Authors<span class="caret"></span> </a>\n<ul class="dropdown-menu dropdown-menu-end">\n<li class="dropdown-item">\n<a href="/authors/john-c-maxwell-quotes" class="bq_on_link_cl" data-xtracl="PA,SubMenu,1">John C. Maxwell</a>\n</li>\n<li class="dropdown-item">\n<a href="/authors/friedrich-nietzsche-quotes" class="bq_on_link_cl" data-xtracl="PA,SubMenu,2">Friedrich Nietzsche</a>\n</li>\n<li class="dropdown-item">\n<a href="/authors/albert-einstein-quotes" class="bq_on_link_cl" data-xtracl="PA,SubMenu,3">Albert Einstein</a>\n</li>\n<li class="dropdown-item">\n<a href="/authors/taylor-swift-quotes" class="bq_on_link_cl" data-xtracl="PA,SubMenu,4">Taylor Swift</a>\n</li>\n<li class="dropdown-item">\n<a href="/authors/ernest-hemingway-quotes" class="bq_on_link_cl" data-xtracl="PA,SubMenu,5">Ernest Hemingway</a>\n</li>\n<li class="dropdown-item">\n<a href="/authors/winston-churchill-quotes" class="bq_on_link_cl" data-xtracl="PA,SubMenu,6">Winston Churchill</a>\n</li>\n<li class="dropdown-item">\n<a href="/authors/matthew-perry-quotes" class="bq_on_link_cl" data-xtracl="PA,SubMenu,7">Matthew Perry</a>\n</li>\n<li class="dropdown-item">\n<a href="/authors/napoleon-bonaparte-quotes" class="bq_on_link_cl" data-xtracl="PA,SubMenu,8">Napoleon Bonaparte</a>\n</li>\n</ul>\n</li>\n<li class="hidden-1col dropdown">\n\n<a href="#" class="dropdown-toggle" data-bs-toggle="dropdown" role="button" aria-haspopup="true" aria-expanded="false" aria-label="Menu"> Menu<span class="caret"></span> </a>\n<ul class="dropdown-menu dropdown-menu-end">\n<li class="dropdown-item">\n<a href="/lists/topics/top-10-wisdom-quotes">Top 10 Wisdom Quotes</a>\n</li>\n<li class="dropdown-item">\n<a href="/citation/topics/wisdom-quotes">Cite this page</a>\n</li>\n<li role="separator" class="divider" aria-hidden="true"></li>\n<li class="dropdown-header">Layout</li>\n<li class="dropdown-item">\n<a href="/topics/wisdom-quotes" onclick="document.cookieUtils.setGridMode();return true;"><i class="bq_grid bq-fa-th"></i>\nGrid</a>\n</li>\n<li class="dropdown-item">\n<a href="/topics/wisdom-quotes" onclick="document.cookieUtils.setListMode();return true;"><i class="bq_list bq-fa-list"></i>\nList</a>\n</li>\n</ul>\n</li>\n<li><a href="/share/fb/u?url=%2Ftopics%2Fwisdom-quotes" aria-label="Share on Facebook" class="sh-fb sh-grey" target="_blank" rel="nofollow"><img src="/st/img/5669107/fa/fb.svg" aria-hidden="true" alt="Share on Facebook" class="bq-fa-sm"></a></li><li><a href="/share/tw/u?url=%2Ftopics%2Fwisdom-quotes&ti=Wisdom+Quotes" aria-label="Share on Twitter" class="sh-tw sh-grey" target="_blank" rel="nofollow"><img src="/st/img/5669107/fa/tw.svg" aria-hidden="true" alt="Share on Twitter" class="bq-fa-sm"></a></li><li><a href="/share/li/u?url=%2Ftopics%2Fwisdom-quotes&ti=Wisdom+Quotes+-+BrainyQuote" aria-label="Share on LinkedIn" class="sh-tw sh-grey" target="_blank" rel="nofollow"><img src="/st/img/5669107/fa/li.svg" aria-hidden="true" alt="Share on LinkedIn" class="bq-fa-sm"></a></li>\n</ul>\n</div>\n</div>\n</div>\n</div>\n<div class="bq-nav-spacer"></div>\n<noscript>\n<div class="enableJs">\n<b>Please enable Javascript</b><br>\nThis site requires Javascript to function properly, please enable it.\n</div>\n</noscript>\n</nav>\n<main>\n<div class="bq-top-spc r_dt"></div>\n<div id="quotesList" class="bq-mg bqcpx grid-layout-hide reflow_body bq_center ql_page">\n<div id="qbcc" class="qbcol-c">\n<div id="qbc1" class="qbcol qbc-spc">\n<div id="pos_1_1" class="grid-item qb clearfix bqQt">\n<div class="qti-listm">\n<a title="view quote" href="/quotes/archimedes_101761?src=t_wisdom" class="oncl_q"> <img id="qimage_101761" src="/photos_tr/en/a/archimedes/101761/archimedes1.jpg" class="bqphtgrid" alt="Give me a lever long enough and a fulcrum on which to place it, and I shall move the world. - Archimedes" width="600" height="315"></a>\n</div>\n<a href="/quotes/archimedes_101761?src=t_wisdom" class="b-qt qt_101761 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nGive me a lever long enough and a fulcrum on which to place it, and I shall move the world.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/archimedes-quotes" class="bq-aut qa_101761 oncl_a" title="view author">Archimedes</a>\n</div>\n<div id="pos_1_2" class="grid-item qb clearfix bqQt">\n<a href="/quotes/socrates_101212?src=t_wisdom" class="b-qt qt_101212 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nThe only true wisdom is in knowing you know nothing.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/socrates-quotes" class="bq-aut qa_101212 oncl_a" title="view author">Socrates</a>\n</div>\n<div id="pos_1_3" class="grid-item qb clearfix bqQt">\n<a href="/quotes/lao_tzu_137141?src=t_wisdom" class="b-qt qt_137141 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nThe journey of a thousand miles begins with one step.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/lao-tzu-quotes" class="bq-aut qa_137141 oncl_a" title="view author">Lao Tzu</a>\n</div>\n<div id="pos_1_4" class="m-ad-brick grid-item m-ad-minh boxy-ad">\n<div id="div-gpt-ad-1418667263920-4_edge" class="mbl_qtbox bq_300x250_edge ">\n<div id="div-gpt-ad-1418667263920-4" class="bq_ad_square_multi bqAdCollapse "></div>\n</div>\n</div>\n<div id="pos_1_5" class="grid-item qb clearfix bqQt">\n<a href="/quotes/napoleon_bonaparte_103585?src=t_wisdom" class="b-qt qt_103585 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nNever interrupt your enemy when he is making a mistake.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/napoleon-bonaparte-quotes" class="bq-aut qa_103585 oncl_a" title="view author">Napoleon Bonaparte</a>\n</div>\n<div id="pos_1_6" class="grid-item qb clearfix bqQt">\n<a href="/quotes/lewis_carroll_165865?src=t_wisdom" class="b-qt qt_165865 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nIf you dont know where you are going, any road will get you there.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/lewis-carroll-quotes" class="bq-aut qa_165865 oncl_a" title="view author">Lewis Carroll</a>\n</div>\n<div id="pos_1_7" class="grid-item qb clearfix bqQt">\n<a href="/quotes/jim_elliot_189244?src=t_wisdom" class="b-qt qt_189244 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nHe is no fool who gives what he cannot keep to gain what he cannot lose.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/jim-elliot-quotes" class="bq-aut qa_189244 oncl_a" title="view author">Jim Elliot</a>\n</div>\n<div id="pos_1_8" class="grid-item qb clearfix bqQt">\n<a href="/quotes/henry_david_thoreau_106041?src=t_wisdom" class="b-qt qt_106041 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nIts not what you look at that matters, its what you see.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/henry-david-thoreau-quotes" class="bq-aut qa_106041 oncl_a" title="view author">Henry David Thoreau</a>\n</div>\n<div id="pos_1_9" class="grid-item qb clearfix bqQt">\n<a href="/quotes/walter_scott_118003?src=t_wisdom" class="b-qt qt_118003 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nO, what a tangled web we weave when first we practise to deceive!\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/walter-scott-quotes" class="bq-aut qa_118003 oncl_a" title="view author">Walter Scott</a>\n</div>\n<div id="pos_1_10" class="grid-item qb clearfix bqQt">\n<div class="qti-listm">\n<a title="view quote" href="/quotes/epictetus_149126?src=t_wisdom" class="oncl_q"> <img id="qimage_149126" src="/photos_tr/en/e/epictetus/149126/epictetus1.jpg" class="bqphtgrid" alt="Its not what happens to you, but how you react to it that matters. - Epictetus" loading="lazy" width="600" height="315"></a>\n</div>\n<a href="/quotes/epictetus_149126?src=t_wisdom" class="b-qt qt_149126 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nIts not what happens to you, but how you react to it that matters.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/epictetus-quotes" class="bq-aut qa_149126 oncl_a" title="view author">Epictetus</a>\n</div>\n<div id="pos_1_11" class="m-ad-brick grid-item m-ad-minh boxy-ad">\n<div id="div-gpt-ad-1418667263920-3-mult_auct2_edge" class="mbl_qtbox bq_300x250_edge ">\n<div id="div-gpt-ad-1418667263920-3-mult_auct2" class="bq_ad_square_multi bqAdCollapse "></div>\n</div>\n</div>\n<div id="pos_1_12" class="grid-item qb clearfix bqQt">\n<a href="/quotes/confucius_131984?src=t_wisdom" class="b-qt qt_131984 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nBy three methods we may learn wisdom: First, by reflection, which is noblest; Second, by imitation, which is easiest; and third by experience, which is the bitterest.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/confucius-quotes" class="bq-aut qa_131984 oncl_a" title="view author">Confucius</a>\n</div>\n<div id="pos_1_13" class="grid-item qb clearfix bqQt">\n<a href="/quotes/nelson_mandela_121685?src=t_wisdom" class="b-qt qt_121685 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nIf you talk to a man in a language he understands, that goes to his head. If you talk to him in his language, that goes to his heart.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/nelson-mandela-quotes" class="bq-aut qa_121685 oncl_a" title="view author">Nelson Mandela</a>\n</div>\n<div id="pos_1_14" class="grid-item qb clearfix bqQt">\n<a href="/quotes/jim_rohn_109882?src=t_wisdom" class="b-qt qt_109882 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nDiscipline is the bridge between goals and accomplishment.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/jim-rohn-quotes" class="bq-aut qa_109882 oncl_a" title="view author">Jim Rohn</a>\n</div>\n<div id="pos_1_15" class="grid-item qb clearfix bqQt">\n<a href="/quotes/william_arthur_ward_110212?src=t_wisdom" class="b-qt qt_110212 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nThe pessimist complains about the wind; the optimist expects it to change; the realist adjusts the sails.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/william-arthur-ward-quotes" class="bq-aut qa_110212 oncl_a" title="view author">William Arthur Ward</a>\n</div>\n<div id="pos_1_16" class="grid-item qb clearfix bqQt">\n<a href="/quotes/pope_paul_vi_120370?src=t_wisdom" class="b-qt qt_120370 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nNever reach out your hand unless youre willing to extend an arm.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/pope-paul-vi-quotes" class="bq-aut qa_120370 oncl_a" title="view author">Pope Paul VI</a>\n</div>\n<div id="pos_1_17" class="grid-item qb clearfix bqQt">\n<a href="/quotes/albert_einstein_100298?src=t_wisdom" class="b-qt qt_100298 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nReality is merely an illusion, albeit a very persistent one.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/albert-einstein-quotes" class="bq-aut qa_100298 oncl_a" title="view author">Albert Einstein</a>\n</div>\n<div id="pos_1_18" class="m-ad-brick grid-item m-ad-minh boxy-ad">\n<div id="div-gpt-ad-1418667263920-2-mult_auct3_edge" class="mbl_qtbox bq_300x250_edge ">\n<div id="div-gpt-ad-1418667263920-2-mult_auct3" class="bq_ad_square_multi bqAdCollapse "></div>\n</div>\n</div>\n<div id="pos_1_19" class="grid-item qb clearfix bqQt">\n<a href="/quotes/johann_wolfgang_von_goeth_161238?src=t_wisdom" class="b-qt qt_161238 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nEverything in the world may be endured except continual prosperity.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/johann-wolfgang-von-goeth-quotes" class="bq-aut qa_161238 oncl_a" title="view author">Johann Wolfgang von Goethe</a>\n</div>\n<div id="pos_1_20" class="grid-item qb clearfix bqQt">\n<div class="qti-listm">\n<a title="view quote" href="/quotes/theodore_roosevelt_109482?src=t_wisdom" class="oncl_q"> <img id="qimage_109482" src="/photos_tr/en/t/theodoreroosevelt/109482/theodoreroosevelt1.jpg" class="bqphtgrid" alt="Nine-tenths of wisdom is being wise in time. - Theodore Roosevelt" loading="lazy" width="600" height="315"></a>\n</div>\n<a href="/quotes/theodore_roosevelt_109482?src=t_wisdom" class="b-qt qt_109482 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nNine-tenths of wisdom is being wise in time.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/theodore-roosevelt-quotes" class="bq-aut qa_109482 oncl_a" title="view author">Theodore Roosevelt</a>\n</div>\n<div id="pos_1_21" class="grid-item qb clearfix bqQt">\n<a href="/quotes/swami_vivekananda_213399?src=t_wisdom" class="b-qt qt_213399 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nWe are what our thoughts have made us; so take care about what you think. Words are secondary. Thoughts live; they travel far.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/swami-vivekananda-quotes" class="bq-aut qa_213399 oncl_a" title="view author">Swami Vivekananda</a>\n</div>\n<div id="pos_1_22" class="grid-item qb clearfix bqQt">\n<a href="/quotes/john_burroughs_120946?src=t_wisdom" class="b-qt qt_120946 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nThe smallest deed is better than the greatest intention.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/john-burroughs-quotes" class="bq-aut qa_120946 oncl_a" title="view author">John Burroughs</a>\n</div>\n<div id="pos_1_23" class="grid-item qb clearfix bqQt">\n<a href="/quotes/george_bernard_shaw_141483?src=t_wisdom" class="b-qt qt_141483 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nBeware of false knowledge; it is more dangerous than ignorance.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/george-bernard-shaw-quotes" class="bq-aut qa_141483 oncl_a" title="view author">George Bernard Shaw</a>\n</div>\n<div id="pos_1_24" class="grid-item qb clearfix bqQt">\n<a href="/quotes/michelangelo_108779?src=t_wisdom" class="b-qt qt_108779 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nThe greater danger for most of us lies not in setting our aim too high and falling short; but in setting our aim too low, and achieving our mark.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/michelangelo-quotes" class="bq-aut qa_108779 oncl_a" title="view author">Michelangelo</a>\n</div>\n<div id="pos_1_25" class="grid-item qb clearfix bqQt">\n<a href="/quotes/john_muir_108391?src=t_wisdom" class="b-qt qt_108391 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nIn every walk with nature one receives far more than he seeks.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/john-muir-quotes" class="bq-aut qa_108391 oncl_a" title="view author">John Muir</a>\n</div>\n<div id="pos_1_26" class="grid-item qb clearfix bqQt">\n<a href="/quotes/will_rogers_104938?src=t_wisdom" class="b-qt qt_104938 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nEven if youre on the right track, youll get run over if you just sit there.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/will-rogers-quotes" class="bq-aut qa_104938 oncl_a" title="view author">Will Rogers</a>\n</div>\n<div id="pos_1_27" class="m-ad-brick grid-item m-ad-minh boxy-ad">\n<div id="div-gpt-ad-1418667263920-1-mult_auct4_edge" class="mbl_qtbox bq_300x250_edge ">\n<div id="div-gpt-ad-1418667263920-1-mult_auct4" class="bq_ad_square_multi bqAdCollapse "></div>\n</div>\n</div>\n<div id="pos_1_28" class="grid-item qb clearfix bqQt">\n<a href="/quotes/ralph_waldo_emerson_106883?src=t_wisdom" class="b-qt qt_106883 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nAdopt the pace of nature: her secret is patience.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/ralph-waldo-emerson-quotes" class="bq-aut qa_106883 oncl_a" title="view author">Ralph Waldo Emerson</a>\n</div>\n<div id="pos_1_29" class="grid-item qb clearfix bqQt">\n<div class="qti-listm">\n<a title="view quote" href="/quotes/lucille_ball_384638?src=t_wisdom" class="oncl_q"> <img id="qimage_384638" src="/photos_tr/en/l/lucilleball/384638/lucilleball1.jpg" class="bqphtgrid" alt="Id rather regret the things Ive done than regret the things I havent done. - Lucille Ball" loading="lazy" width="600" height="315"></a>\n</div>\n<a href="/quotes/lucille_ball_384638?src=t_wisdom" class="b-qt qt_384638 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nId rather regret the things Ive done than regret the things I havent done.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/lucille-ball-quotes" class="bq-aut qa_384638 oncl_a" title="view author">Lucille Ball</a>\n</div>\n<div id="pos_1_30" class="grid-item qb clearfix bqQt">\n<a href="/quotes/audrey_hepburn_394440?src=t_wisdom" class="b-qt qt_394440 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nFor beautiful eyes, look for the good in others; for beautiful lips, speak only words of kindness; and for poise, walk with the knowledge that you are never alone.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/audrey-hepburn-quotes" class="bq-aut qa_394440 oncl_a" title="view author">Audrey Hepburn</a>\n</div>\n<div id="pos_1_31" class="grid-item qb clearfix bqQt">\n<a href="/quotes/john_wooden_386606?src=t_wisdom" class="b-qt qt_386606 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nTalent is God given. Be humble. Fame is man-given. Be grateful. Conceit is self-given. Be careful.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/john-wooden-quotes" class="bq-aut qa_386606 oncl_a" title="view author">John Wooden</a>\n</div>\n<div id="pos_1_32" class="grid-item qb clearfix bqQt">\n<a href="/quotes/martin_luther_king_jr_387472?src=t_wisdom" class="b-qt qt_387472 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nThat old law about an eye for an eye leaves everybody blind. The time is always right to do the right thing.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/martin-luther-king-jr-quotes" class="bq-aut qa_387472 oncl_a" title="view author">Martin Luther King, Jr.</a>\n</div>\n<div id="pos_1_33" class="grid-item qb clearfix bqQt">\n<a href="/quotes/oliver_wendell_holmes_sr_108494?src=t_wisdom" class="b-qt qt_108494 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nThe young man knows the rules, but the old man knows the exceptions.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/oliver-wendell-holmes-sr-quotes" class="bq-aut qa_108494 oncl_a" title="view author">Oliver Wendell Holmes, Sr.</a>\n</div>\n<div id="pos_1_34" class="grid-item qb clearfix bqQt">\n<a href="/quotes/yogi_berra_380870?src=t_wisdom" class="b-qt qt_380870 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nIf the world were perfect, it wouldnt be.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/yogi-berra-quotes" class="bq-aut qa_380870 oncl_a" title="view author">Yogi Berra</a>\n</div>\n<div id="pos_1_35" class="grid-item qb clearfix bqQt">\n<a href="/quotes/carl_jung_114802?src=t_wisdom" class="b-qt qt_114802 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nEverything that irritates us about others can lead us to an understanding of ourselves.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/carl-jung-quotes" class="bq-aut qa_114802 oncl_a" title="view author">Carl Jung</a>\n</div>\n<div id="pos_1_36" class="m-ad-brick grid-item m-ad-minh boxy-ad">\n<div id="div-gpt-ad-1418667263920-4-mult_auct5_edge" class="mbl_qtbox bq_300x250_edge ">\n<div id="div-gpt-ad-1418667263920-4-mult_auct5" class="bq_ad_square_multi bqAdCollapse "></div>\n</div>\n</div>\n<div id="pos_1_37" class="grid-item qb clearfix bqQt">\n<a href="/quotes/john_c_maxwell_391398?src=t_wisdom" class="b-qt qt_391398 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nA man must be big enough to admit his mistakes, smart enough to profit from them, and strong enough to correct them.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/john-c-maxwell-quotes" class="bq-aut qa_391398 oncl_a" title="view author">John C. Maxwell</a>\n</div>\n<div id="pos_1_38" class="grid-item qb clearfix bqQt">\n<a href="/quotes/w_clement_stone_193778?src=t_wisdom" class="b-qt qt_193778 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nYou are a product of your environment. So choose the environment that will best develop you toward your objective. Analyze your life in terms of its environment. Are the things around you helping you toward success - or are they holding you back?\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/w-clement-stone-quotes" class="bq-aut qa_193778 oncl_a" title="view author">W. Clement Stone</a>\n</div>\n<div id="pos_1_39" class="grid-item qb clearfix bqQt">\n<div class="qti-listm">\n<a title="view quote" href="/quotes/satchel_paige_390217?src=t_wisdom" class="oncl_q"> <img id="qimage_390217" src="/photos_tr/en/s/satchelpaige/390217/satchelpaige1.jpg" class="bqphtgrid" alt="Work like you dont need the money. Love like youve never been hurt. Dance like nobodys watching. - Satchel Paige" loading="lazy" width="600" height="315"></a>\n</div>\n<a href="/quotes/satchel_paige_390217?src=t_wisdom" class="b-qt qt_390217 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nWork like you dont need the money. Love like youve never been hurt. Dance like nobodys watching.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/satchel-paige-quotes" class="bq-aut qa_390217 oncl_a" title="view author">Satchel Paige</a>\n</div>\n<div id="pos_1_40" class="grid-item qb clearfix bqQt">\n<a href="/quotes/thomas_a_edison_109928?src=t_wisdom" class="b-qt qt_109928 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nGenius is one percent inspiration and ninety-nine percent perspiration.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/thomas-a-edison-quotes" class="bq-aut qa_109928 oncl_a" title="view author">Thomas A. Edison</a>\n</div>\n<div id="pos_1_41" class="grid-item qb clearfix bqQt">\n<a href="/quotes/dr_seuss_597903?src=t_wisdom" class="b-qt qt_597903 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nStep with care and great tact, and remember that Lifes a Great Balancing Act.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/dr-seuss-quotes" class="bq-aut qa_597903 oncl_a" title="view author">Dr. Seuss</a>\n</div>\n<div id="pos_1_42" class="grid-item qb clearfix bqQt">\n<a href="/quotes/henry_ford_101486?src=t_wisdom" class="b-qt qt_101486 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nObstacles are those frightful things you see when you take your eyes off your goal.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/henry-ford-quotes" class="bq-aut qa_101486 oncl_a" title="view author">Henry Ford</a>\n</div>\n<div id="pos_1_43" class="grid-item qb clearfix bqQt">\n<a href="/quotes/walt_disney_385581?src=t_wisdom" class="b-qt qt_385581 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nThe more you like yourself, the less you are like anyone else, which makes you unique.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/walt-disney-quotes" class="bq-aut qa_385581 oncl_a" title="view author">Walt Disney</a>\n</div>\n<div id="pos_1_44" class="grid-item qb clearfix bqQt">\n<a href="/quotes/jeanpaul_sartre_417004?src=t_wisdom" class="b-qt qt_417004 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nCommitment is an act, not a word.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/jean-paul-sartre-quotes" class="bq-aut qa_417004 oncl_a" title="view author">Jean-Paul Sartre</a>\n</div>\n<div id="pos_1_45" class="grid-item qb clearfix bqQt">\n<a href="/quotes/anton_chekhov_119058?src=t_wisdom" class="b-qt qt_119058 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nKnowledge is of no value unless you put it into practice.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/anton-chekhov-quotes" class="bq-aut qa_119058 oncl_a" title="view author">Anton Chekhov</a>\n</div>\n<div id="pos_1_46" class="grid-item qb clearfix bqQt">\n<a href="/quotes/margaret_thatcher_131837?src=t_wisdom" class="b-qt qt_131837 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nIf you set out to be liked, you would be prepared to compromise on anything at any time, and you would achieve nothing.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/margaret-thatcher-quotes" class="bq-aut qa_131837 oncl_a" title="view author">Margaret Thatcher</a>\n</div>\n<div id="pos_1_47" class="m-ad-brick grid-item m-ad-minh boxy-ad">\n<div id="div-gpt-ad-1418667263920-3-mult_auct6_edge" class="mbl_qtbox bq_300x250_edge ">\n<div id="div-gpt-ad-1418667263920-3-mult_auct6" class="bq_ad_square_multi bqAdCollapse "></div>\n</div>\n</div>\n<div id="pos_1_48" class="grid-item qb clearfix bqQt">\n<div class="qti-listm">\n<a title="view quote" href="/quotes/henry_van_dyke_391213?src=t_wisdom" class="oncl_q"> <img id="qimage_391213" src="/photos_tr/en/h/henryvandyke/391213/henryvandyke1.jpg" class="bqphtgrid" alt="Use what talents you possess; the woods would be very silent if no birds sang there except those that sang best. - Henry Van Dyke" loading="lazy" width="600" height="315"></a>\n</div>\n<a href="/quotes/henry_van_dyke_391213?src=t_wisdom" class="b-qt qt_391213 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nUse what talents you possess; the woods would be very silent if no birds sang there except those that sang best.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/henry-van-dyke-quotes" class="bq-aut qa_391213 oncl_a" title="view author">Henry Van Dyke</a>\n</div>\n<div id="pos_1_49" class="grid-item qb clearfix bqQt">\n<a href="/quotes/lord_byron_150380?src=t_wisdom" class="b-qt qt_150380 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nSorrow is knowledge, those that know the most must mourn the deepest, the tree of knowledge is not the tree of life.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/lord-byron-quotes" class="bq-aut qa_150380 oncl_a" title="view author">Lord Byron</a>\n</div>\n<div id="pos_1_50" class="grid-item qb clearfix bqQt">\n<a href="/quotes/rabindranath_tagore_385179?src=t_wisdom" class="b-qt qt_385179 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nEverything comes to us that belongs to us if we create the capacity to receive it.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/rabindranath-tagore-quotes" class="bq-aut qa_385179 oncl_a" title="view author">Rabindranath Tagore</a>\n</div>\n<div id="pos_1_51" class="grid-item qb clearfix bqQt">\n<a href="/quotes/soren_kierkegaard_152222?src=t_wisdom" class="b-qt qt_152222 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nOnce you label me you negate me.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/soren-kierkegaard-quotes" class="bq-aut qa_152222 oncl_a" title="view author">Soren Kierkegaard</a>\n</div>\n<div id="pos_1_52" class="grid-item qb clearfix bqQt">\n<a href="/quotes/judy_garland_104276?src=t_wisdom" class="b-qt qt_104276 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nAlways be a first-rate version of yourself, instead of a second-rate version of somebody else.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/judy-garland-quotes" class="bq-aut qa_104276 oncl_a" title="view author">Judy Garland</a>\n</div>\n<div id="pos_1_53" class="grid-item qb clearfix bqQt">\n<a href="/quotes/michael_jordan_129404?src=t_wisdom" class="b-qt qt_129404 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nIf youre trying to achieve, there will be roadblocks. Ive had them; everybody has had them. But obstacles dont have to stop you. If you run into a wall, dont turn around and give up. Figure out how to climb it, go through it, or work around it.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/michael-jordan-quotes" class="bq-aut qa_129404 oncl_a" title="view author">Michael Jordan</a>\n</div>\n<div id="pos_1_54" class="grid-item qb clearfix bqQt">\n<a href="/quotes/elizabeth_kenny_114982?src=t_wisdom" class="b-qt qt_114982 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nIts better to be a lion for a day than a sheep all your life.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/elizabeth-kenny-quotes" class="bq-aut qa_114982 oncl_a" title="view author">Elizabeth Kenny</a>\n</div>\n<div id="pos_1_55" class="grid-item qb clearfix bqQt">\n<a href="/quotes/amelia_earhart_120932?src=t_wisdom" class="b-qt qt_120932 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nNever interrupt someone doing what you said couldnt be done.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/amelia-earhart-quotes" class="bq-aut qa_120932 oncl_a" title="view author">Amelia Earhart</a>\n</div>\n<div id="pos_1_56" class="grid-item qb clearfix bqQt">\n<div class="qti-listm">\n<a title="view quote" href="/quotes/george_eliot_122525?src=t_wisdom" class="oncl_q"> <img id="qimage_122525" src="/photos_tr/en/g/georgeeliot/122525/georgeeliot1.jpg" class="bqphtgrid" alt="Our deeds determine us, as much as we determine our deeds. - George Eliot" loading="lazy" width="600" height="315"></a>\n</div>\n<a href="/quotes/george_eliot_122525?src=t_wisdom" class="b-qt qt_122525 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nOur deeds determine us, as much as we determine our deeds.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/george-eliot-quotes" class="bq-aut qa_122525 oncl_a" title="view author">George Eliot</a>\n</div>\n<div id="pos_1_57" class="grid-item qb clearfix bqQt">\n<a href="/quotes/george_s_patton_106027?src=t_wisdom" class="b-qt qt_106027 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nNever tell people how to do things. Tell them what to do and they will surprise you with their ingenuity.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/george-s-patton-quotes" class="bq-aut qa_106027 oncl_a" title="view author">George S. Patton</a>\n</div>\n<div id="pos_1_58" class="m-ad-brick grid-item m-ad-minh boxy-ad">\n<div id="div-gpt-ad-1418667263920-2-mult_auct7_edge" class="mbl_qtbox bq_300x250_edge ">\n<div id="div-gpt-ad-1418667263920-2-mult_auct7" class="bq_ad_square_multi bqAdCollapse "></div>\n</div>\n</div>\n<div id="pos_1_59" class="grid-item qb clearfix bqQt">\n<a href="/quotes/benjamin_franklin_151625?src=t_wisdom" class="b-qt qt_151625 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nHonesty is the best policy.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/benjamin-franklin-quotes" class="bq-aut qa_151625 oncl_a" title="view author">Benjamin Franklin</a>\n</div>\n<div id="pos_1_60" class="grid-item qb clearfix bqQt">\n<a href="/quotes/joan_rivers_386753?src=t_wisdom" class="b-qt qt_386753 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nYesterday is history, tomorrow is a mystery, today is Gods gift, thats why we call it the present.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/joan-rivers-quotes" class="bq-aut qa_386753 oncl_a" title="view author">Joan Rivers</a>\n</div>\n<div id="pos_1_61" class="grid-item qb clearfix bqQt">\n<a href="/quotes/robert_h_schuller_120883?src=t_wisdom" class="b-qt qt_120883 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nSpectacular achievement is always preceded by unspectacular preparation.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/robert-h-schuller-quotes" class="bq-aut qa_120883 oncl_a" title="view author">Robert H. Schuller</a>\n</div>\n<div id="pos_1_62" class="grid-item qb clearfix bqQt">\n<a href="/quotes/khalil_gibran_108029?src=t_wisdom" class="b-qt qt_108029 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nThe teacher who is indeed wise does not bid you to enter the house of his wisdom but rather leads you to the threshold of your mind.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/khalil-gibran-quotes" class="bq-aut qa_108029 oncl_a" title="view author">Khalil Gibran</a>\n</div>\n<div id="pos_1_63" class="grid-item qb clearfix bqQt">\n<a href="/quotes/lucius_annaeus_seneca_108502?src=t_wisdom" class="b-qt qt_108502 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nNo man was ever wise by chance.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/lucius-annaeus-seneca-quotes" class="bq-aut qa_108502 oncl_a" title="view author">Lucius Annaeus Seneca</a>\n</div>\n<div id="pos_1_64" class="grid-item qb clearfix bqQt">\n<a href="/quotes/aldous_huxley_145888?src=t_wisdom" class="b-qt qt_145888 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nExperience is not what happens to you; its what you do with what happens to you.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/aldous-huxley-quotes" class="bq-aut qa_145888 oncl_a" title="view author">Aldous Huxley</a>\n</div>\n<div id="pos_1_65" class="grid-item qb clearfix bqQt">\n<div class="qti-listm">\n<a title="view quote" href="/quotes/arnold_schwarzenegger_146576?src=t_wisdom" class="oncl_q"> <img id="qimage_146576" src="/photos_tr/en/a/arnoldschwarzenegger/146576/arnoldschwarzenegger1.jpg" class="bqphtgrid" alt="Start wide, expand further, and never look back. - Arnold Schwarzenegger" loading="lazy" width="600" height="315"></a>\n</div>\n<a href="/quotes/arnold_schwarzenegger_146576?src=t_wisdom" class="b-qt qt_146576 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nStart wide, expand further, and never look back.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/arnold-schwarzenegger-quotes" class="bq-aut qa_146576 oncl_a" title="view author">Arnold Schwarzenegger</a>\n</div>\n<div id="pos_1_66" class="grid-item qb clearfix bqQt">\n<a href="/quotes/francis_bacon_110048?src=t_wisdom" class="b-qt qt_110048 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nIt is impossible to love and to be wise.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/francis-bacon-quotes" class="bq-aut qa_110048 oncl_a" title="view author">Francis Bacon</a>\n</div>\n<div id="pos_1_67" class="grid-item qb clearfix bqQt">\n<a href="/quotes/john_f_kennedy_110220?src=t_wisdom" class="b-qt qt_110220 oncl_q" title="view quote">\n<div style="display: flex;justify-content: space-between">\nThe time to repair the roof is when the sun is shining.\n<img src="/st/img/5669107/fa/chv-r.svg" alt="Share this Quote" class="bq-qb-chv">\n</div>\n</a>\n<a href="/authors/john-f-kennedy-quotes" class="bq-aut qa_110220 oncl_a" title="view author">John F. Kennedy</a>\n</div>\n</div>\n<div id="qbc2" class="qbcol qbc-spc"></div>\n<div id="qbc3" class="qbcol qbc-spc"></div>\n<div id="qbc4" class="qbcol qbc-spc"></div>\n<div id="qbc5" class="qbcol qbc-spc"></div>\n<div id="qbc6" class="qbcol qbc-spc"></div>\n</div>\n</div>\n<div class="infScrollFooter">\n<div class="bq_s hideInfScroll bq_pageNumbersCont">\n<ul class="pagination bq_pageNumbers pagination-centered pagination-sm">\n<li class="page-item disabled"><span class="page-link">Prev</span></li>\n<li class="page-item active"><span class="page-link">1</span></li>\n<li class="page-item"><a href="/topics/wisdom-quotes_2" class="page-link">2</a></li>\n<li class="page-item"><a href="/topics/wisdom-quotes_3" class="page-link">3</a></li>\n<li class="page-item"><a href="/topics/wisdom-quotes_4" class="page-link">4</a></li>\n<li class="page-item disabled"><span class="page-link">..</span></li>\n<li class="page-item"><a href="/topics/wisdom-quotes_17" class="page-link">17</a></li>\n<li class="page-item"><a href="/topics/wisdom-quotes_2" class="page-link">Next</a>\n</li>\n</ul>\n</div>\n<div class="bq_footer_ad">\n<div id="div-gpt-ad-1418667263920-3-mb_footer_edge" class="mbl_qtbox bq_300x250_edge ">\n<div id="div-gpt-ad-1418667263920-3-mb_footer" class="bq_ad_square_multi bqAdCollapse "></div>\n</div>\n<div style="height:20px;"></div>\n</div>\n<div class="container-fluid" style="max-width:1350px;margin-top:25px;margin-bottom:25px;">\n<div class="row">\n<div class="col-12 block-style">\n<div class="bq_s row">\n<h2>Recommended Topics</h2>\n<div class="col-lg-3 col-6 block-sm-holder">\n<a href="/topics/life-quotes" class="block-sm bq_on_link_cl" data-xtracl="RT,Wisdom,1"><span class="link-name">Life<br/>\n<span>Quotes</span>\n</span></a>\n</div>\n<div class="col-lg-3 col-6 block-sm-holder">\n<a href="/topics/motivational-quotes" class="block-sm bq_on_link_cl" data-xtracl="RT,Wisdom,2"><span class="link-name">Motivational<br/>\n<span>Quotes</span>\n</span></a>\n</div>\n<div class="col-lg-3 col-6 block-sm-holder">\n<a href="/topics/change-quotes" class="block-sm bq_on_link_cl" data-xtracl="RT,Wisdom,3"><span class="link-name">Change<br/>\n<span>Quotes</span>\n</span></a>\n</div>\n<div class="col-lg-3 col-6 block-sm-holder">\n<a href="/topics/inspirational-quotes" class="block-sm bq_on_link_cl" data-xtracl="RT,Wisdom,4"><span class="link-name">Inspirational<br/>\n<span>Quotes</span>\n</span></a>\n</div>\n<div class="col-lg-3 col-6 block-sm-holder">\n<a href="/topics/education-quotes" class="block-sm bq_on_link_cl" data-xtracl="RT,Wisdom,5"><span class="link-name">Education<br/>\n<span>Quotes</span>\n</span></a>\n</div>\n<div class="col-lg-3 col-6 block-sm-holder">\n<a href="/topics/forgiveness-quotes" class="block-sm bq_on_link_cl" data-xtracl="RT,Wisdom,6"><span class="link-name">Forgiveness<br/>\n<span>Quotes</span>\n</span></a>\n</div>\n<div class="col-lg-3 col-6 block-sm-holder">\n<a href="/topics/good-quotes" class="block-sm bq_on_link_cl" data-xtracl="RT,Wisdom,7"><span class="link-name">Good<br/>\n<span>Quotes</span>\n</span></a>\n</div>\n<div class="col-lg-3 col-6 block-sm-holder">\n<a href="/topics/smile-quotes" class="block-sm bq_on_link_cl" data-xtracl="RT,Wisdom,8"><span class="link-name">Smile<br/>\n<span>Quotes</span>\n</span></a>\n</div>\n</div>\n</div>\n</div>\n</div>\n</div>\n</main>\n<div id="mbl_stcky_c" role="region" aria-label="Page footer advertisement" class="mbl_stcky_sh r_mb_med ">\n<div class="mbl_stcky_w">\n<div class="mbl_stcky_mw">\n<div id="bq_mobile_anchor_placement-mstck_edge" class=" ">\n<div id="bq_mobile_anchor_placement-mstck" class="bq_ad_mobile_anchor bqAdCollapse "></div>\n</div>\n</div>\n</div>\n</div>\n<footer>\n<div id="bqBotNav">\n<div class="bq_bot_nav bq-no-print">\n<div class="container-fluid" style="max-width: 900px">\n<div class="row">\n<div class="col-sm-6 col-md-3 col-12">\n<div style="float:left; position:absolute; top:-25px;" class="bq_pinweel">\n<img src="/st/img/5669107/pinwheel.gif" alt="BrainyQuote" class="bqPw">\n</div>\n<div style="height:75px"></div>\n<div class="bq_s">\n<ul class="footer-follow-icons">\n<li><a href="https://www.facebook.com/BrainyQuote" aria-label="Follow us on Facebook" target="_blank" rel="noreferrer"><img src="/st/img/5669107/fa/fb.svg" alt="Share on Facebook" class="bq-fa-full">\n</a></li>\n<li><a href="https://twitter.com/brainyquote" aria-label="Follow us on Twitter" target="_blank" rel="noreferrer"><img src="/st/img/5669107/fa/tw.svg" alt="Share on Twitter" class="bq-fa-full"></a></li>\n<li><a href="https://pinterest.com/brainyquote" aria-label="Follow us on Pinterest" target="_blank" rel="noreferrer"><img src="/st/img/5669107/fa/pinterest.svg" alt="Share on Pinterest" class="bq-fa-full"></a></li>\n<li class="follow-padding-fix"><a href="https://instagram.com/brainyquote" aria-label="Follow us on Instagram" target="_blank" rel="noreferrer"><img src="/st/img/5669107/fa/ig.svg" alt="Share on Instagram" class="bq-fa-full"></a></li>\n</ul>\n<div style="clear:both"></div>\n</div>\n</div>\n<div class="col-sm-6 col-md-3 col-12">\n<div class="bq_s">\nBrainyQuote has been providing inspirational quotes since 2001 to our worldwide community.\n</div>\n<div class="bq_s">\n<h3>Quote Of The Day Feeds</h3>\n<div class="bqLn"><a href="/feeds/todays_quote">Javascript and RSS feeds</a></div>\n<div class="bqLn"><a href="/feeds/wordpress_plugin">WordPress plugin</a></div>\n<div class="bqLn"><a href="/feeds/quote_of_the_day_email">Quote of the Day Email</a></div>\n</div>\n</div>\n<div class="col-sm-6 col-md-2 col-12">\n<div class="bq_s">\n<h3>Site</h3>\n<div class="bqLn"><a href="/">Home</a></div>\n<div class="bqLn"><a href="/authors">Authors</a></div>\n<div class="bqLn"><a href="/topics">Topics</a></div>\n<div class="bqLn"><a href="/quote_of_the_day">Quote Of The Day</a></div>\n<div class="bqLn"><a href="/top_100_quotes">Top 100 Quotes</a></div>\n<div class="bqLn"><a href="/profession/">Professions</a></div>\n<div class="bqLn"><a href="/birthdays/">Birthdays</a></div>\n</div>\n</div>\n<div class="col-sm-6 col-md-2 col-12">\n<div class="bq_s">\n<h3>About</h3>\n<div class="bqLn"><a href="/about/">About Us</a></div>\n<div class="bqLn"><a href="/about/contact_us">Contact Us</a></div>\n<div class="bqLn"><a href="/about/privacy">Privacy</a></div>\n<div class="bqLn"><a href="/about/terms">Terms</a></div>\n</div>\n</div>\n<div class="col-sm-6 col-md-2 col-12">\n<div class="bq_s">\n<h3>Apps</h3>\n<div class="bqLn"><a href="/apps/">iOS app</a></div>\n<br>\n</div>\n</div>\n</div>\n</div>\n</div>\n<div class="copyLangFooter langSelLink">\n<div class="container">\n<div class="row ">\n<div class="col-sm-5 col-md-5">\n<div class="ftr_nav" style="vertical-align:center; padding-left: 15px"><a href="/about/copyright" style="vertical-align:top">Copyright</a>\n&#169; 2001 - 2023 BrainyQuote\n</div>\n</div>\n<div id="usp-c" class="col-sm-5 col-md-5 usp-footer-hide">\n<div id="usp-footer" class="ftr_nav" style="vertical-align:center; padding-left: 15px;">\n<a href="/ccpa">Do Not Sell My Info</a>\n</div>\n</div>\n</div>\n</div>\n</div>\n<script type="application/ld+json">\n{\n"@context": "http://schema.org",\n"@type": "WebSite",\n"url": "https://www.brainyquote.com/",\n"name": "BrainyQuote",\n"alternateName": "BrainyQuote - Famous Quotes",\n"potentialAction": {\n"@type": "SearchAction",\n"target": "https://www.brainyquote.com/search_results?q={search_term}",\n"query-input": "required name=search_term"\n}\n}\n</script>\n<script type="application/ld+json">\n{\n"@context": "http://schema.org",\n"@type": "BreadcrumbList",\n"itemListElement": [\n{\n"item": {\n"name": "Quote Topics",\n"@id": "https://www.brainyquote.com/topics"\n},\n"@type": "ListItem",\n"position": 1\n},\n{\n"item": {\n"name": "Wisdom Quotes",\n"@id": "https://www.brainyquote.com/topics/wisdom-quotes"\n},\n"@type": "ListItem",\n"position": 2\n}\n]\n}\n</script>\n</div>\n</footer>\n<script>(function(){var js = "window[\'__CF$cv$params\']={r:\'826b809c2d2836c7\',t:\'MTcwMDA5MzMzNy4xMzUwMDA=\'};_cpo=document.createElement(\'script\');_cpo.nonce=\'\',_cpo.src=\'/cdn-cgi/challenge-platform/scripts/jsd/main.js\',document.getElementsByTagName(\'head\')[0].appendChild(_cpo);";var _0xh = document.createElement(\'iframe\');_0xh.height = 1;_0xh.width = 1;_0xh.style.position = \'absolute\';_0xh.style.top = 0;_0xh.style.left = 0;_0xh.style.border = \'none\';_0xh.style.visibility = \'hidden\';document.body.appendChild(_0xh);function handler() {var _0xi = _0xh.contentDocument || _0xh.contentWindow.document;if (_0xi) {var _0xj = _0xi.createElement(\'script\');_0xj.innerHTML = js;_0xi.getElementsByTagName(\'head\')[0].appendChild(_0xj);}}if (document.readyState !== \'loading\') {handler();} else if (window.addEventListener) {document.addEventListener(\'DOMContentLoaded\', handler);} else {var prev = document.onreadystatechange || function () {};document.onreadystatechange = function (e) {prev(e);if (document.readyState !== \'loading\') {document.onreadystatechange = prev;handler();}};}})();</script></body>\n</html>\n'
    #html = html.decode('utf-8')
    #
    ##print(html)
    ##
    ##return
    #
    ##if not os.path.isdir( FOLDER_NAME ):
    ##    os.makedirs( FOLDER_NAME )
    #if not os.path.isdir( DATA_FOLDER_NAME ):
    #    os.makedirs( DATA_FOLDER_NAME )
    #        
    #print( "Parsing..." )
    #parser = QuotesLinksParser()
    #parser.feed(html)
    #
    ##img_parser = ImageParser()
    ##print( parser.desc )
    #for link in parser.links:
    #    if VERBOSE:
    #        print( "- Parsed " + link.pretty() )
    #        
    #    #quote_title = link.url.replace("/quotes/","")
    #    #try:
    #    #    quote_title = quote_title[:quote_title.index("?src=")]
    #    #except:
    #    #    print( "Skipping", quote_title)
    #    #    continue
    #    ##if VERBOSE:
    #    #print( quote_title )
    #    ###TODO: If an image in images starts with quote_title, skip!
    #    ##skip = False
    #    #skip_data = False
    #    ##for image in os.listdir( FOLDER_NAME ):
    #    ##    if image.startswith( quote_title ):
    #    ##        if VERBOSE:
    #    ##            print( "SKIP! (%s)" % image )
    #    ##        skip = True
    #    #for data in os.listdir( DATA_FOLDER_NAME ):
    #    #    if data.startswith( quote_title ):
    #    #        print( "SKIP DATA! (%s) " % data )
    #    #        skip_data = True
    #    #        
    #    ##if not skip or not skip_data:
    #    ##    content = get_content2( "https://apod.nasa.gov/apod/" + link.url )
    #    ##    img_parser.clear()
    #    ##    img_parser.feed( content )        
    #    ##        
    #    ##if not skip:
    #    ##    #if VERBOSE:
    #    ##    #TODO: Uncomment!
    #    ##    print( img_parser.images )
    #    ##    for img in img_parser.images:
    #    ##        get_file( "https://apod.nasa.gov/apod/" + img, prefix=quote_title + "_" + link.date + "_")# folder = link.date?
    #    ##        #print( img )
    #    ##
    #    ##        
    #    ##if not skip_data:
    #    #    #print( img_parser.desc, len(img_parser.images) )
    #    #    #TODOOOOOOOOOO use link 
    #    #    #write_file( link=link, name=quote_title, date=link.date, urls = img_parser.images )
    #    #    #DATA_FOLDER_NAME
    #    
    #parser.authors = sorted(parser.authors, key=lambda x:x.caption)
    #
    #for author in parser.authors:
    #    if VERBOSE:
    #        print( "Parsed author: " + author.pretty() )
    #   
    #            
    ##get_file( "https://yt3.ggpht.com/-TdCBbRamnvs/AAAAAAAAAAI/AAAAAAAAAAA/z0HO1Olmsqw/s88-c-k-no-mo-rj-c0xffffff/photo.jpg", 
    ##    prefix="2017")

            
     
# ---------------------------------------------------------------------------------------------------        
def get_descriptions(steps):
    import random
    lines_image_desc = [
        "A cascading waterfall roars down a moss-covered cliff, sending mist swirling into the air.",
        "A powerful waterfall crashes down onto jagged rocks, creating a thunderous sound that echoes through the canyon.",
        "A hidden waterfall trickles down a rocky slope, creating a peaceful and serene atmosphere.",
        "A misty waterfall shrouded in a perpetual cloud of spray, with rainbows dancing in the mist.",
        "A majestic waterfall cascading into a turquoise lagoon surrounded by lush green palm trees.",
        "A frozen waterfall in winter, transformed into a crystal palace with icicles hanging from the rocks.",
        "A waterfall flowing through a colorful autumn forest, with leaves painted in shades of red, orange, and yellow.",
        "A waterfall cascading down a volcanic rock face, with molten lava flowing nearby.",
        "A waterfall flowing upwards, defying gravity in a dreamlike scene.",
        "A waterfall hidden behind a curtain of glowing mushrooms in a magical forest.",
        "A waterfall cascading into a bottomless pit, shrouded in mystery and mist.",
        "A waterfall flowing through a network of ancient ruins, overgrown with vines and vegetation.",
        "A cascading waterfall plunges through a lush rainforest, sunlight filtering through the leaves to create a dappled effect on the rushing water.", 
        "A powerful waterfall roars down a rocky cliff face in a remote mountain range, mist rising from the churning pool below.",
        "A hidden waterfall tucked away in a mossy grotto, surrounded by ferns and blooming wildflowers.",
        "A majestic waterfall in Iceland, framed by black volcanic rock and fed by glacial meltwater.",
        "A wide, curtain-like waterfall thunders over a horseshoe-shaped crest, creating a rainbow in the spray.",
        "A multi-tiered waterfall cascades down a series of rocky steps, each pool reflecting the vibrant blue sky.",
        "A plunge waterfall plummets straight down a sheer cliff face, disappearing into a cloud of mist below.",
        "A tiered waterfall with a hidden cave behind the cascading water, accessible only by a narrow path.",
        "A frozen waterfall in winter, the rushing water transformed into glittering ice sculptures.",
        "A waterfall during a vibrant autumn season, surrounded by fiery red and orange leaves.",
        "A waterfall bathed in the warm glow of a summer sunset, creating a magical scene.",
        "A waterfall swollen with spring runoff, churning with a powerful flow of water.",
        "A bioluminescent waterfall at night, the cascading water glowing with an ethereal blue light.",
        "A waterfall flowing into a crystal-clear pool, surrounded by smooth, white sand.",
        "A waterfall hidden behind a veil of thick vines, accessible only by following a secret path.",
        "A waterfall with an ancient temple built into the cliff face beside it.",
        "Waterfall"
        ]

    # Cap lines_image_desc at the length of steps
    lines_image_desc = lines_image_desc[:len(steps)]

    # Extend lines_image_desc with random strings if needed
    while len(lines_image_desc) < len(steps):
        random_string = random.choice(lines_image_desc)
        lines_image_desc.append(random_string)

    return lines_image_desc

    
# --------------------------------------------------------------------------------------------------- 
def seconds_to_mmss(seconds):
  """
  This function converts a number of seconds to a string in the format "mm:ss".

  Args:
      seconds: The number of seconds to convert (int).

  Returns:
      A string in the format "mm:ss" representing the minutes and seconds.
  """
  # Ensure seconds is a non-negative integer
  if not isinstance(seconds, int) or seconds < 0:
    raise ValueError("Input must be a non-negative integer")

  # Get minutes and remaining seconds
  minutes, remaining_seconds = divmod(seconds, 60)

  # Format minutes and seconds with leading zeros
  minutes_str = f"{minutes:02d}"
  seconds_str = f"{remaining_seconds:02d}"

  return f"{minutes_str}:{seconds_str}"
    

# --------------------------------------------------------------------------------------------------- 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fast", action="store_true", default=False, help="fast rendering")
parser.add_argument("--half", action="store_true", default=False, help="Half rendering (screws up the fast flag)")
parser.add_argument("--font", default=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX )#FONT_HERSHEY_COMPLEX
parser.add_argument("-x", "--width", default=1920)#1080)
parser.add_argument("-y", "--height", default=1080)#1920)
parser.add_argument("-d", "--videoDuration", type=int, default=1200, help="in seconds")
parser.add_argument("-s", "--showImages", action="store_true", default=False)

args = parser.parse_args()
if args.fast:
    #output_size = (216,384)
    args.width = int(args.width/10)
    args.height = int(args.height/10)
    args.font = cv2.FONT_HERSHEY_SIMPLEX  
    args.videoDuration = int(args.videoDuration/10)
    FONT_SCALING = 100
    show_images_raw = False
if args.half:
    args.width = int(args.width/2)
    args.height = int(args.height/2)
    args.font = cv2.FONT_HERSHEY_SIMPLEX  
    #args.videoDuration = int(args.videoDuration/2)
    FONT_SCALING = 100
    show_images_raw = False


# ---------------------------------------------------------------------------------------------------
#main(args)
nb_steps = max( args.videoDuration, 79-2)#42
from_hz = 639 
to_hz=0.98275#713#0.975 #0.95
mp3_name, steps, duration = generate.generate_mp3(duration=(args.videoDuration+(nb_steps+2)*SILENCE_TIME_SEC), from_hz=from_hz, to_hz=to_hz, nb_steps=nb_steps, do_plot=False, initial_steps=[396])
print(f"name: {mp3_name}, {duration}s/step, steps: {steps}")

print("duration0:",args.videoDuration)
print("duration1:",args.videoDuration+(nb_steps+2)*SILENCE_TIME_SEC)
print("durations:", [duration-SILENCE_TIME_SEC] * len(steps))
print("duration3:", sum([duration-SILENCE_TIME_SEC] * len(steps)))
print("duration4:", sum([duration-SILENCE_TIME_SEC] * len(steps)) + len(steps)*SILENCE_TIME_SEC )

title="BrownNoiseFading"
video_directory = "BrownNoiseOutput"

# TODO: Same len() as steps
lines = []


#TODOOOOOOOOO: https://stackoverflow.com/questions/71962098/python-opencv-puttext-show-non-ascii-unicode-utf-character-symbols
current_time = 0
for i, s in enumerate(steps):
    if IS_FRENCH:
        if i == 0:#256
            lines.append("Voici du Bruit Brun ("+str(s)+"Hz): On commence plus aigu, pour terminer plus grave")
        #elif i == 1:#32
        #    lines.append(str(steps[-1])+"Hz sera le plus grave")
        elif i == 1:#2:#992Hz?
            lines.append("Revenons plus aigu ("+str(s)+"Hz). La descente commence, pour vous calmer et vous endormir")
        else:
            lines.append(str(s)+"Hz")
    else:
        if i == 0:#256
            lines.append("This is Brown Noise at "+str(s)+"Hz. It will start high and go to lower frequencies during the video")
        #elif i == 1:#32
        #    lines.append(str(steps[-1])+"Hz: The video will end at this frequency. At this level, the sound becomes lower and deeper")
        elif i == 1:#2:#992Hz?
            lines.append("Returning to a higher frequency, at "+str(s)+"Hz. It will gradually go deeper, to help you relax and fall asleep")
        else:
            lines.append(str(s)+"Hz")
    
            
    print(seconds_to_mmss(math.ceil(current_time))+ " " + lines[-1])
    current_time += duration# + SILENCE_TIME_SEC

       

lines_image_desc = get_descriptions(steps)
#print(lines_image_desc)

#nb_frames_fade = int((SILENCE_TIME_SEC*2)*fps)
nb_frames_fade = int(1.5*fps)

render_everything(video_directory, lines_image_desc, True, title, [duration-SILENCE_TIME_SEC] * len(steps),
    nb_frames_fade, mp3_name)
