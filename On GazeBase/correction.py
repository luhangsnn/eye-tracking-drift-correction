def generate_fixations_lr_skip_regression(aois_with_tokens):
    
    fixations = []
    word_count = 0
    skip_count = 0
    regress_count = 0
    
    aoi_list = aois_with_tokens.values.tolist()
    
    index = 0
    
    while index < len(aoi_list):
        x, y, width, height, token = aoi_list[index][2], aoi_list[index][3], aoi_list[index][4], aoi_list[index][5], aoi_list[index][7]
        
        word_count += 1
        
        fixation_x = x + width / 2 + random.randint(-10, 10)
        fixation_y = y + height / 2 + random.randint(-10, 10)

        last_skipped = False

        # skipping
        if len(token) < 5 and random.random() < 0.3:
            skip_count += 1 # fixations.append([fixation_x, fixation_y])
            last_skipped = True
        else:
            fixations.append([fixation_x, fixation_y, len(token) * 50])
            last_skipped = False
        
        # regressions    
        if  random.random() > 0.96:
            index -= random.randint(1, 10)

            if index < 0:
                index = 0

            regress_count += 1
        
        index += 1
            
    
    skip_probability = skip_count / word_count
    
    return fixations

# function to generate fixations at the center of each word
def generate_fixations_center(aois_with_tokens):
    
    fixations = []
    
    for index, row in aois_with_tokens.iterrows():
        x, y, width, height, token = row['x'], row['y'], row['width'], row['height'], row['token']
        
        fixation_x = x + width / 2
        fixation_y = y + height / 2
        
        fixations.append([fixation_x, fixation_y, len(token) * 50])
        
    return fixations


def generate_fixations_left(aois_with_tokens):
    
    fixations = []
    
    for index, row in aois_with_tokens.iterrows():
        x, y, width, height, token = row['x'], row['y'], row['width'], row['height'], row['token']
        
        fixation_x = x + width / 3
        fixation_y = y + height / 2
        
        fixations.append([fixation_x, fixation_y, len(token) * 50])
        
    return fixations


def generate_fixations_left_skip(aois_with_tokens):
    
    fixations = []
    word_count = 0
    skip_count = 0
    
    for index, row in aois_with_tokens.iterrows():
        x, y, width, height, token = row['x'], row['y'], row['width'], row['height'], row['token']
        
        word_count += 1
        
        fixation_x = x + width / 3
        fixation_y = y + height / 2

        if len(token) < 4 and random.random() > 0.7:
            skip_count += 1 # fixations.append([fixation_x, fixation_y])
        else:
            fixations.append([fixation_x, fixation_y, len(token) * 50])
    
    print(skip_count / word_count)
    return fixations


def generate_fixations_left_skip(aois_with_tokens, skip_probability):
    
    fixations = []
    word_count = 0
    skip_count = 0
    
    for index, row in aois_with_tokens.iterrows():
        x, y, width, height, token = row['x'], row['y'], row['width'], row['height'], row['token']
        
        word_count += 1
        
        fixation_x = x + width / 3
        fixation_y = y + height / 2

        if random.random() < skip_probability:
            skip_count += 1 # fixations.append([fixation_x, fixation_y])
        else:
            fixations.append([fixation_x, fixation_y, len(token) * 50])
    
    #print(skip_count / word_count)
    return fixations



def generate_fixations_left_skip_regression(aois_with_tokens):
    
    fixations = []
    word_count = 0
    skip_count = 0
    regress_count = 0
    
    aoi_list = aois_with_tokens.values.tolist()
    
    index = 0
    
    while index < len(aoi_list):
        x, y, width, height, token = aoi_list[index][2], aoi_list[index][3], aoi_list[index][4], aoi_list[index][5], aoi_list[index][7]
        
        word_count += 1
        
        fixation_x = x + width / 3 + random.randint(-10, 10)
        fixation_y = y + height / 2 + random.randint(-10, 10)

        last_skipped = False

        # skipping
        if len(token) < 5 and random.random() < 0.3:
            skip_count += 1 # fixations.append([fixation_x, fixation_y])
            last_skipped = True
        else:
            fixations.append([fixation_x, fixation_y, len(token) * 50])
            last_skipped = False
        
        # regressions    
        if  random.random() > 0.96:
            index -= random.randint(1, 10)

            if index < 0:
                index = 0

            regress_count += 1
        
        index += 1
            
    
    skip_probability = skip_count / word_count
    
    return fixations


def generate_fixed_regression(aois_with_tokens):
    
    fixations = [[196, 169],
    [319, 169],
    [414, 169],
    [481, 169],
    [550, 169],
    [674, 169],
    [778, 169],
    [826, 169],
    [890, 169],
    [948, 169],
    [996, 169],
    [208, 219],
    [319, 219],
    [430, 219],
    [598, 219],
    [717, 219],
    [785, 219],
    [901, 219],
    [1005, 219],
    [163, 268],
    [244, 268],
    [346, 268],
    [444, 268],
    [517, 268],
        [598, 219],
        [717, 219],
        [785, 219],
        [901, 219],
        [1005, 219],
    [565, 268],
    [666, 268],
    [798, 268],
    [881, 268],
    [945, 268],
    [1013, 268],
    [183, 318],
    [273, 318],
    [373, 318],
        [666, 268],
        [798, 268],
    [487, 318],
    [595, 318],
    [679, 318],
    [761, 318],
    [842, 318],
    [901, 318],
    [980, 318],
    [178, 368],
    [252, 368],
    [325, 368],
    [436, 368],
    [579, 368],
    [689, 368],
    [822, 368],
    [955, 368],
    [1019, 368],
    [215, 417],
    [372, 417],
    [491, 417],
        [980, 318],
        [178, 368],
    [579, 417],
    [660, 417],
    [766, 417]]
    
    
    return fixations


def generate_fixations_left_regression(aois_with_tokens, regression_probability):
    
    fixations = []
    word_count = 0
    regress_count = 0
    
    aoi_list = aois_with_tokens.values.tolist()
    
    index = 0
    
    while index < len(aoi_list):
        x, y, width, height, token = aoi_list[index][2], aoi_list[index][3], aoi_list[index][4], aoi_list[index][5], aoi_list[index][7]
        
        word_count += 1
        
        fixation_x = x + width / 3  + random.randint(-10, 10)
        fixation_y = y + height / 2 + random.randint(-10, 10)

        fixations.append([fixation_x, fixation_y, len(token) * 50])
        
        if  random.random() < regression_probability/5:
            index -= random.randint(1, 10)
            if index < 0:
            	index = 0
            regress_count += 1
        
        index += 1
    
    return fixations



def error_offset(x_offset, y_offset, fixations):
    '''creates error to move fixations (shift in dissertation)'''
    
    results = []

    for fix in fixations:

        x, y = fix[0], fix[1]
        results.append([x + x_offset, y + y_offset, fix[2]])
    
    return results

# noise
import random

def error_noise(y_noise_probability, y_noise, duration_noise, fixations):
    '''creates a random error moving a percentage of fixations '''
    
    results = []
    
    for fix in fixations:

        x, y, duration = fix[0], fix[1], fix[2]

        # should be 0.1 for %10
        duration_error = int(duration * duration_noise)

        duration += random.randint(-duration_error, duration_error)

        if duration < 0:
            duration *= -1
        
        if random.random() < y_noise_probability:
            results.append([x, y + random.randint(-y_noise, y_noise), duration])
        else:
            results.append([x, y, fix[2]])
    
    return results

# shift

def error_shift(y_shift_factor, line_ys, fixations):
    '''creates error moving fixations above or below line progressively'''

    results = []
    
    for fix in fixations:

        x, y = fix[0], fix[1]
        
        distance_from_first_line = abs(y - line_ys[0])
        
        if distance_from_first_line > 40:
            results.append([x, y + ((distance_from_first_line % 55) * y_shift_factor), fix[2]])
        else:
            results.append([x, y, fix[2]])
        
    return results


# droop

def error_droop(droop_factor, fixations):
    """creates droop error"""
    
    results = []
    
    first_x = fixations[0][0]
    
    for fix in fixations:

        x, y = fix[0], fix[1]

        results.append([x , y + ((x - first_x)/100 * droop_factor), fix[2]])
        
    return results

from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
import numpy as np


def fix_to_img(Image_file, fixations):
    im = Image.open(Image_file)
    draw = ImageDraw.Draw(im, 'RGB')

    if len(fixations[0]) == 3:
        x0, y0, duration = fixations[0]
    else:
        x0, y0 = fixations[0]
        
    for fixation in fixations:
        
        if len(fixations[0]) == 3:
            duration = fixation[2]
            if 5 * (duration / 100) < 5:
                r = 3
            else:
                r = 5 * (duration / 100)
        else:
            r = 8
        x = fixation[0]
        y = fixation[1]

        bound = (x - r, y - r, x + r, y + r)
        outline_color = (50, 255, 0, 0)
        fill_color = (50, 255, 0, 220)
        draw.ellipse(bound, fill=fill_color, outline=outline_color)

        bound = (x0, y0, x, y)
        line_color = (255, 155, 0, 155)
        penwidth = 2
        draw.line(bound, fill=line_color, width=5)

        x0, y0 = x, y
    
    return im

def fix_to_img_general(Image_file, fixations, aois_with_tokens):
    """Private method that draws the fixation, also allow user to draw eye movement order

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        a Draw object imposed on the image

    draw_number : bool
        whether user wants to draw the eye movement number
    """

    base = Image.open(Image_file)
    im = Image.new("RGB", base.size, (255, 255, 255))
    draw = ImageDraw.Draw(im, "RGB")

    if len(fixations[0]) == 3:
        x0, y0, duration = fixations[0]
    else:
        x0, y0 = fixations[0]
    
    for index, row in aois_with_tokens.iterrows():
        x, y, width, height, token = row['x'], row['y'], row['width'], row['height'], row['token']

        bound = (x , y, x+width, y+height)
        outline_color = (189, 195, 199, 1)
        fill_color = (189, 195, 199, 1)
        draw.rectangle(bound, fill = fill_color, outline = outline_color)

    for fixation in fixations:
        
        if len(fixations[0]) == 3:
            duration = fixation[2]
            if 5 * (duration / 100) < 5:
                r = 3
            else:
                r = 5 * (duration / 100)
        else:
            r = 8
        x = fixation[0]
        y = fixation[1]

        bound = (x - r, y - r, x + r, y + r)
        outline_color = (0, 0, 0)
        fill_color = (0, 0, 0)
        draw.ellipse(bound, fill=fill_color, outline=outline_color)

        bound = (x0, y0, x, y)
        line_color = (0, 0, 0)
        penwidth = 2
        draw.line(bound, fill=line_color, width=5)

        x0, y0 = x, y
    
    return im

def draw_fixation(Image_file, fixations):
    """Private method that draws the fixation, also allow user to draw eye movement order

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        a Draw object imposed on the image

    draw_number : bool
        whether user wants to draw the eye movement number
    """

    im = Image.open(Image_file)
    draw = ImageDraw.Draw(im, 'RGBA')

    if len(fixations[0]) == 3:
        x0, y0, duration = fixations[0]
    else:
        x0, y0 = fixations[0]

    for fixation in fixations:
        
        if len(fixations[0]) == 3:
            duration = fixation[2]
            if 5 * (duration / 100) < 5:
                r = 3
            else:
                r = 5 * (duration / 100)
        else:
            r = 8
        x = fixation[0]
        y = fixation[1]

        bound = (x - r, y - r, x + r, y + r)
        outline_color = (50, 255, 0, 0)
        fill_color = (50, 255, 0, 220)
        draw.ellipse(bound, fill=fill_color, outline=outline_color)

        bound = (x0, y0, x, y)
        line_color = (255, 155, 0, 155)
        penwidth = 2
        draw.line(bound, fill=line_color, width=5)

        x0, y0 = x, y

    plt.figure(figsize=(17, 15))
    plt.imshow(np.asarray(im), interpolation='nearest')


def draw_correction(Image_file, fixations, match_list):
    """Private method that draws the fixation, also allow user to draw eye movement order

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        a Draw object imposed on the image

    draw_number : bool
        whether user wants to draw the eye movement number
    """

    im = Image.open(Image_file)
    draw = ImageDraw.Draw(im, 'RGBA')

    if len(fixations[0]) == 3:
        x0, y0, duration = fixations[0]
    else:
        x0, y0 = fixations[0]

    for index, fixation in enumerate(fixations):
        
        if len(fixations[0]) == 3:
            duration = fixation[2]
            if 5 * (duration / 100) < 5:
                r = 3
            else:
                 r = 5 * (duration / 100)
        else:
            r = 8

        x = fixation[0]
        y = fixation[1]

        bound = (x - r, y - r, x + r, y + r)
        outline_color = (50, 255, 0, 0)
        
        if match_list[index] == 1:
        	fill_color = (50, 255, 0, 220)
        else:
        	fill_color = (255, 55, 0, 220)

        draw.ellipse(bound, fill=fill_color, outline=outline_color)

        bound = (x0, y0, x, y)
        line_color = (255, 155, 0, 155)
        penwidth = 2
        draw.line(bound, fill=line_color, width=5)

        # text_bound = (x + random.randint(-10, 10), y + random.randint(-10, 10))
        # text_color = (0, 0, 0, 225)
        # font = ImageFont.truetype("arial.ttf", 20)
        # draw.text(text_bound, str(index), fill=text_color,font=font)

        x0, y0 = x, y

    plt.figure(figsize=(17, 15))
    plt.imshow(np.asarray(im), interpolation='nearest')


def find_lines_Y(aois):
    ''' returns a list of line Ys '''
    
    results = []
    
    for index, row in aois.iterrows():
        y, height = row['y'], row['height']
        
        if y + height / 2 not in results:
            results.append(y + height / 2)
            
    return results



def find_word_centers(aois):
    ''' returns a list of word centers '''
    
    results = []
    
    for index, row in aois.iterrows():
        x, y, height, width = row['x'], row['y'], row['height'], row['width']
        
        center = [int(x + width // 2), int(y + height // 2)]
        
        if center not in results:
            results.append(center)
            
    return results


def find_word_centers_and_duration(aois):
    ''' returns a list of word centers '''
    
    results = []
    
    for index, row in aois.iterrows():
        x, y, height, width, token = row['x'], row['y'], row['height'], row['width'], row['token']
        
        center = [int(x + width // 2), int(y + height // 2), len(token) * 50]

        if center not in results:
            results.append(center)
    
    #print(results)
    return results


def find_word_centers_and_EZ_duration(aois):
    ''' returns a list of word centers '''
    
    results = []
    
    for index, row in aois.iterrows():
        x, y, height, width, token = row['x'], row['y'], row['height'], row['width'], row['token']
        
        duration = row['GD']

        center = [int(x + width // 2), int(y + height // 2), int(duration)]

        if center not in results:
            results.append(center)
    
    #print(results)
    return results


def overlap(fix, AOI):
    """checks if fixation is within AOI"""
    
    box_x = AOI.x
    box_y = AOI.y
    box_w = AOI.width
    box_h = AOI.height

    if fix[0] >= box_x and fix[0] <= box_x + box_w \
    and fix[1] >= box_y and fix[1] <= box_y + box_h:
        return True
    
    else:
        
        return False
    
    
def correction_quality(aois, original_fixations, corrected_fixations):
    
    match = 0
    total_fixations = len(original_fixations)
    results = [0] * total_fixations
    
    for index, fix in enumerate(original_fixations):
        
        for _, row  in aois.iterrows():
            
            if overlap(fix, row) and overlap(corrected_fixations[index], row):
                match += 1
                results[index] = 1
                
    return match / total_fixations, results


import drift_algorithms as algo

def my_attach(fixations, line_ys):
    
    results = fixations.copy()
    
    for fix in results:
        
        min_distance = 999999
        line_y = 0
        
        for line in line_ys:
            
            if abs(fix[1] - line) < min_distance:
                min_distance = abs(fix[1] - line)
                line_y = line
        
        fix[1] = line_y
        
    return results

def split_regressions(fixation_data):
    without_regs = []
    regs = []
    regs_indexs = []

    last_fixation = fixation_data[0]
    without_regs.append(last_fixation)
    index = 1
    stuck = False

    while index < len(fixation_data):

        if ((fixation_data[index][0] < last_fixation[0] and fixation_data[index][1] < last_fixation[1])
        or fixation_data[index][1] < last_fixation[1]) and (last_fixation[1] - fixation_data[index][1] > 0):

                regs.append(last_fixation)

                while index < len(fixation_data) and last_fixation[1] - fixation_data[index][1] > 0 and last_fixation[0] - fixation_data[index][0] <= 0:
                    regs.append(fixation_data[index])
                    regs_indexs.append(index)
                    index += 1
                
                if not stuck:
                    #print("index", index, "stuck!", "fix len:", len(fixation_data))
                    stuck = True
                    regs_indexs.append(index)
                    continue

        else:
            without_regs.append(fixation_data[index])

        last_fixation = fixation_data[index]
        index += 1
        stuck = False
    
    return without_regs, regs, regs_indexs


def remove_regs(fixations, regs_indexs):
    warp_input = []

    for index in range(len(fixations)):

        if index not in regs_indexs:
            warp_input.append(fixations[index])

    return warp_input


def add_regs(corrected, original, regs_indexs):
    
    results = corrected.copy()

    # uniform length rows
    for index in range(len(results)):
        results[index] = [results[index][0], results[index][1]]
    
    for index in regs_indexs:
        if index < len(original):
            results.insert(index, [int(original[index][0]), int(original[index][1])] )
    
    return results
    

def warp_regs(error_test, line_ys, word_centers):
    
    # remove regressions
    fixation_data = my_attach(error_test.copy(), line_ys)
    without_regs, regs, regs_indexs = split_regressions(fixation_data)
    warp_input =  remove_regs(error_test.copy(), regs_indexs)
    
    # apply basic warp
    np_array = np.array(warp_input.copy(), dtype=int)
    warp_correction = algo.warp(np_array, word_centers)
    warp_correction = warp_tolist()

    # add regression back to warp_correction
    result = add_regs(warp_correction, fixation_data.copy(), regs_indexs)
    #print(result)

    # iron out regressions with chain
    np_array = np.array(result.copy(), dtype=int)

    # if np_array.shape[1] == 3:
    #     durations = np.delete(np_array, 0, 1)
    #     durations = np.delete(durations, 0, 1)
    #     np_array = np.delete(np_array, 2, 1)

    return algo.chain(np_array, line_ys)
    #return add_regs(warp_correction, error_test.copy(), regs_indexs)


from PIL import Image, ImageChops

def trim_image(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def prep_image(img_path, fixations, aoi_with_tokens, input_x, input_y):
    im = fix_to_img_general(img_path, fixations, aoi_with_tokens)
    im_trim = trim_image(im)
    im_small = im_trim.resize((input_y, input_x))
    im_small_array = np.asarray(im_small)

    return im_small_array


def predict_error(img_path, fixations, aoi_with_tokens, input_x, input_y, model):
    '''
    predit the error category for each trial
    0: noise
    1: shift
    2: droop
    3: offset
    4: no error
    '''
    # print(input_x, input_y)
    im_small_array = prep_image(img_path, fixations, aoi_with_tokens, input_x, input_y)
    pred_percent = model.predict(im_small_array.reshape(1, input_x, input_y, 3))
    prediction = np.argmax(pred_percent, axis=1) # 2-level list 
    
    return prediction[0]


def validate_prediction(prediction, y_label):
    '''
    return 1 if the prediction is correct, 0 otherwise
    '''
    
    if prediction == y_label:
        return 1
    else:
        return 0


def make_error_trial (error_type, factor, aois_with_tokens):
    '''
    generate synthetic reading trials and add a specific error to the trial
    '''
    synth_fixations = generate_fixations_lr_skip_regression(aois_with_tokens)
        
    if error_type == 0: # noise
        error_trial = error_noise(factor/10.0, random.randint(0, 50), factor/10.0, synth_fixations) 
    elif error_type == 1: # shift
        line_ys = np.array(synth_fixations)[:, 1]
        error_trial = error_shift(factor/10.0, line_ys, synth_fixations) 

    elif error_type == 2: # droop
        error_trial = error_droop(factor, synth_fixations)  
    
    elif error_type == 3: # offset 
        error_trial = error_offset(factor*2.0, factor*2.0, synth_fixations)
    elif error_type == 4: # no error
        error_trial = synth_fixations  

    else: # wrong entry
        print("Error: wrong entry; there is no error category")
        return
    return error_trial

def add_random_error(error_type, original_trial, factor):
    '''
    add a specific error to a given reading trial
    '''
    if error_type == 0: # noise
        error_trial = error_noise(factor/10.0, random.randint(0, 50), factor/10.0, original_trial) 
    elif error_type == 1: # shift
        line_ys = np.array(original_trial)[:, 1]
        error_trial = error_shift(factor/10.0, line_ys, original_trial) 

    elif error_type == 2: # droop
        error_trial = error_droop(factor, original_trial)  
    
    elif error_type == 3: # offset 
        error_trial = error_offset(factor*2.0, factor*2.0, original_trial)
    elif error_type == 4: # no error
        error_trial = original_trial

    else: # wrong entry
        print("Error: wrong entry; there is no error category")
        return
    return error_trial


def apply_correction(algo_name, trial, line_ys, duration_word_centers, word_centers):
    if algo_name == "attach":
        return algo.attach(trial, line_ys)
    elif algo_name == "chain":
        return algo.chain(trial, line_ys)
    elif algo_name == "cluster":
        return algo.cluster(trial, line_ys)
    elif algo_name == "compare":
        return algo.compare(trial, duration_word_centers)
    elif algo_name == "merge":
        return algo.merge(trial, line_ys)
    elif algo_name == "regress":
        return algo.regress(trial, line_ys)
    elif algo_name == "segment":
        return algo.segment(trial, line_ys)
    elif algo_name == "split":
        return algo.split(trial, line_ys)
    elif algo_name == "stretch":
        return algo.stretch(trial, line_ys)
    elif algo_name == "warp":
        durations = np.delete(trial, 0, 1)
        durations = np.delete(durations, 0, 1)
        trial = np.delete(trial, 2, 1)
        return algo.warp(trial, word_centers)
    else:
        print("Error: wrong entry; there is no algorithm")
        exit()