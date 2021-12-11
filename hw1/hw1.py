from PIL import Image
import os
import math
import numpy as np


key_words = ['xyrgb', 'xyc', 'png', 'pngs', 'trig', 'linec', 'lineg', 'tric', 'xyrgba', 'trica', 'polyec', 'polynz', 'fann', 'stripn']

# return the x value corresponding to the y value on ths given line
def find_y_on_line(p, q, y):
    a = q[1] - p[1]
    b = p[0] - q[0]
    c = a * p[0] + b * p[1]
    # line: ax+by=c
    return (c-b*y)/a


def create_line_color(point1, point2):
    points_to_fill = []
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    reverse = 0

    # assume x is longer; if not, switch x and y
    # assume x1 is smaller than y1
    #going from small x to large x
    if abs(x1-x2) < abs(y1-y2):
        reverse = 1
        y1 = point1[0]
        x1 = point1[1]
        y2 = point2[0]
        x2 = point2[1]

    if x1 > x2:
        x_temp = x2
        y_temp = y2
        x2 = x1
        y2 = y1
        x1 = x_temp
        y1 = y_temp


    slope = (y2-y1)/(x2-x1)
    cur_x = math.ceil(x1)
    delta_x =cur_x - x1
    delta_y = delta_x * slope
    cur_y = y1 + delta_y


    tot_points = math.floor(x2) - cur_x +1
    if not reverse:
        start_color = dict_vertices[(x1, y1)]
        end_color = dict_vertices[(x2, y2)]        

    if reverse:  
        start_color = dict_vertices[(y1, x1)]
        end_color = dict_vertices[(y2, x2)]
    delta_color = np.subtract(end_color, start_color)
    color_slope = np.divide(delta_color, (x2-x1))
    cur_color = np.add(start_color, np.multiply(delta_x, color_slope))
    # print(delta_color

        
    if tot_points == 1:
        if reverse:
            points_to_fill.append((math.floor(cur_y + 0.5), cur_x))
            dict_vertices[(math.floor(cur_y + 0.5), cur_x)] = tuple(cur_color)

        else:          
            points_to_fill.append((cur_x, math.floor(cur_y + 0.5)))
            dict_vertices[(cur_x, math.floor(cur_y + 0.5))] = tuple(cur_color)
        return points_to_fill
        

    while (cur_x < x2):
        if reverse:
            points_to_fill.append((math.floor(cur_y + 0.5), cur_x))
            dict_vertices[(math.floor(cur_y + 0.5), cur_x)] = tuple(cur_color)

        else:          
            points_to_fill.append((cur_x, math.floor(cur_y + 0.5)))
            dict_vertices[(cur_x, math.floor(cur_y + 0.5))] = tuple(cur_color)

        cur_color += np.multiply(delta_color, 1/(x2-x1))

        cur_x += 1
        cur_y += slope
    return points_to_fill




def create_line_color_step_y(point1, point2):
    points_to_fill = []
    y1 = point1[0]
    x1 = point1[1]
    y2 = point2[0]
    x2 = point2[1]

    # assume x is longer; if not, switch x and y
    # assume x1 is smaller than y1
    #going from small x to large x
    if x1 > x2:
        x_temp = x2
        y_temp = y2
        x2 = x1
        y2 = y1
        x1 = x_temp
        y1 = y_temp


    start_color = dict_vertices[(y1, x1)]
    end_color = dict_vertices[(y2, x2)]
    delta_color = np.subtract(end_color, start_color)
    cur_x = math.ceil(x1)
    delta_x =cur_x - x1

    
    if cur_x != math.ceil(x2):
        slope = (y2-y1)/(x2-x1)
        delta_y = delta_x * slope
        cur_y = y1 + delta_y

        color_slope = np.divide(delta_color, (x2-x1))
        cur_color = np.add(start_color, np.multiply(delta_x, color_slope))


        while (cur_x < x2):
            points_to_fill.append((cur_y, cur_x))
            dict_vertices[(cur_y, cur_x)] = tuple(cur_color)


            cur_color += np.multiply(delta_color, 1/(x2-x1))

            cur_x += 1
            cur_y += slope
    return points_to_fill



def even_odd_color(x, y, list_tuples):
    is_in = False
    num_edge = len(list_tuples)
    j = num_edge - 1
    for i in range(num_edge):
        # if is the corner, color it
        if (x == list_tuples[i][0]) and (y == list_tuples[i][1]):    
            return True

        # if the y value is in between two end points
        if ((list_tuples[j][1] > y) != (list_tuples[i][1] > y)):
            slope = (x-list_tuples[i][0])*(list_tuples[j][1]-list_tuples[i][1])
            slope -= (list_tuples[j][0]-list_tuples[i][0])*(y-list_tuples[i][1])
            # check if a point is on the boundary
            if slope == 0:
                return True

            if (slope < 0) != (list_tuples[j][1] < list_tuples[i][1]):
                is_in = not is_in
        j = i

    return is_in


def non_zero_color(x, y, list_tuples):
    is_in = False
    num_edge = len(list_tuples)
    j = num_edge - 1
    for i in range(num_edge):
        # if is the corner, color it
        if (x == list_tuples[i][0]) and (y == list_tuples[i][1]):    
            return True

        # if the y value is in between two end points
        if ((list_tuples[j][1] > y) != (list_tuples[i][1] > y)):
            slope = (x-list_tuples[i][0])*(list_tuples[j][1]-list_tuples[i][1])
            slope -= (list_tuples[j][0]-list_tuples[i][0])*(y-list_tuples[i][1])
            # check if a point is on the boundary
            if slope == 0:
                return True

            if (slope < 0) != (list_tuples[j][1] < list_tuples[i][1]):
                is_in = not is_in
        j = i

    return is_in



with open ('implemented.txt') as files_needed:
    files_all = files_needed.readlines()
    files_all = [f.rstrip() for f in files_all]


    for filename in files_all:
        potential_overlap = {}

        with open(filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines if line.rstrip() != '']
            lines = [line for line in lines if line.split()[0] in key_words]

        mega_info = lines[0].split()
        width = int(mega_info[1])
        height = int(mega_info[2])


        if mega_info[0] == 'png':
            list_vertices = []
            dict_vertices = {}
            filename = mega_info[3]

            image = Image.new("RGBA", (width, height), (0,0,0,0))
            for line in lines[1:]:

                info = line.split()
                op = info[0]

                if op == 'xyrgb':
                    list_vertices.append((float(info[1]),float(info[2]),int(info[3]), int(info[4]), int(info[5]), 255))
                    dict_vertices[(float(info[1]),float(info[2]))] = (int(info[3]), int(info[4]), int(info[5]), 255)

                elif op == 'xyc':
                    color = info[3][1:]
                    list_vertices.append((float(info[1]),float(info[2]),int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), 255))
                    dict_vertices[(float(info[1]),float(info[2]))] = (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), 255)

                elif op == 'xyrgba':
                    list_vertices.append((float(info[1]),float(info[2]),int(info[3]), int(info[4]), int(info[5]), int(info[6])))
                    dict_vertices[(float(info[1]),float(info[2]))] = (int(info[3]), int(info[4]), int(info[5]), int(info[6]))    
                

                #draw a line
                elif op == 'linec':
                    points_to_fill = []
                    color = info[3][1:]
                    if int(info[1]) < 0:
                        point1 = list_vertices[int(info[1])]
                    else:
                        point1 = list_vertices[int(info[1])-1]
                    if int(info[2]) < 0:
                        point2 = list_vertices[int(info[2])]
                    else:
                        point2 = list_vertices[int(info[2])-1]

                    dict_vertices[point1[0:2]] = (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), 255)
                    dict_vertices[point2[0:2]] = (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), 255)
                   

                    points_to_fill = create_line_color(point1, point2)
                    for point in points_to_fill:
                        r, g, b, alpha = dict_vertices[(int(point[0]),int(point[1]))]
                        image.im.putpixel((int(point[0]),int(point[1])), (int(r), int(g), int(b), int(alpha)))



                elif op == 'lineg':
                    if int(info[1]) < 0:
                        point1 = list_vertices[int(info[1])]
                    else:
                        point1 = list_vertices[int(info[1])-1]
                    if int(info[2]) < 0:
                        point2 = list_vertices[int(info[2])]
                    else:
                        point2 = list_vertices[int(info[2])-1]

                    points_to_fill = []
                    
                    points_to_fill = create_line_color(point1, point2)
                    for point in points_to_fill:
                        r, g, b, alpha = dict_vertices[(int(point[0]),int(point[1]))]
                        if (int(point[0]),int(point[1])) in potential_overlap.keys():
                            old_r, old_g, old_b, old_alpha = potential_overlap[(int(point[0]),int(point[1]))]
                        
                            # the new ver is on top, correspond to alpha a
                            alpha /= 255
                            old_alpha /= 255

                            new_alpha = alpha + old_alpha * (1-alpha)
                            new_r = (r*alpha + old_r*old_alpha*(1-alpha))/new_alpha
                            new_g = (g*alpha + old_g*old_alpha*(1-alpha))/new_alpha
                            new_b = (b*alpha + old_b*old_alpha*(1-alpha))/new_alpha
                            potential_overlap[(int(point[0]),int(point[1]))] = (new_r,new_g,new_b,new_alpha*255)
                            image.im.putpixel((int(point[0]),int(point[1])), (int(new_r), int(new_g), int(new_b), int(new_alpha*255)))

                        else:
                            potential_overlap[(int(point[0]),int(point[1]))] = (r,g,b,alpha)
                            image.im.putpixel((int(point[0]),int(point[1])), (int(r), int(g), int(b), int(alpha)))


                elif op == 'tric':
                    points_to_fill_12 = []
                    points_to_fill_13 = []
                    points_to_fill_23 = []
                    color = info[4][1:]

                    if int(info[1]) < 0:
                        point1 = list_vertices[int(info[1])]
                    else:
                        point1 = list_vertices[int(info[1])-1]
                    if int(info[2]) < 0:
                        point2 = list_vertices[int(info[2])]
                    else:
                        point2 = list_vertices[int(info[2])-1]
                    if int(info[3]) < 0:
                        point3 = list_vertices[int(info[3])]
                    else:
                        point3 = list_vertices[int(info[3])-1]

                    dict_vertices[point1[0:2]] = (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), 255)
                    dict_vertices[point2[0:2]] = (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), 255)
                    dict_vertices[point3[0:2]] = (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), 255)


                    points_to_fill_12  = create_line_color_step_y(point1, point2)
                    points_to_fill_13 = create_line_color_step_y(point1, point3)
                    points_to_fill_23 = create_line_color_step_y(point2, point3)


                    all_y = list([a[1] for a in points_to_fill_12] +[a[1] for a in points_to_fill_13] + [a[1] for a in points_to_fill_23])
                    all_y = list(set(all_y))
                    all_y.sort()

                    for y in all_y:
                        all_x = list([a[0] for a in points_to_fill_12 if a[1] == y] +[a[0] for a in points_to_fill_13 if a[1] == y] + [a[0] for a in points_to_fill_23 if a[1] == y])

                        all_x = [x for x in all_x if x is not None]
  
                        start_x = min(all_x)
                        end_x = max(all_x)
                        if math.ceil(start_x) < end_x:

                                points_to_fill = create_line_color((start_x, y), (end_x, y))
                                for point in points_to_fill:
                                    r, g, b, alpha = dict_vertices[(int(point[0]),int(point[1]))]
                                    image.im.putpixel((int(point[0]),int(point[1])), (int(r), int(g), int(b), int(alpha)))



                elif op == 'trica':
                    points_to_fill_12 = []
                    points_to_fill_13 = []
                    points_to_fill_23 = []
                    color = info[4][1:]


                    if int(info[1]) < 0:
                        point1 = list_vertices[int(info[1])]
                    else:
                        point1 = list_vertices[int(info[1])-1]
                    if int(info[2]) < 0:
                        point2 = list_vertices[int(info[2])]
                    else:
                        point2 = list_vertices[int(info[2])-1]
                    if int(info[3]) < 0:
                        point3 = list_vertices[int(info[3])]
                    else:
                        point3 = list_vertices[int(info[3])-1]

                    dict_vertices[point1[0:2]] = (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), int(color[6:8], 16))
                    dict_vertices[point2[0:2]] = (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), int(color[6:8], 16))
                    dict_vertices[point3[0:2]] = (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), int(color[6:8], 16))


                    points_to_fill_12 = create_line_color_step_y(point1, point2)
                    points_to_fill_13 = create_line_color_step_y(point1, point3)
                    points_to_fill_23 = create_line_color_step_y(point2, point3)


                    all_y = list([a[1] for a in points_to_fill_12] +[a[1] for a in points_to_fill_13] + [a[1] for a in points_to_fill_23])
                    all_y = list(set(all_y))
                    all_y.sort()

                    for y in all_y:
                        all_x = list([a[0] for a in points_to_fill_12 if a[1] == y] +[a[0] for a in points_to_fill_13 if a[1] == y] + [a[0] for a in points_to_fill_23 if a[1] == y])
                        all_x = [x for x in all_x if x is not None]
                        start_x = min(all_x)
                        end_x = max(all_x)
                        if math.ceil(start_x) < end_x:

                                points_to_fill = create_line_color((start_x, y), (end_x, y))
                                for point in points_to_fill:
                                    r, g, b, alpha = dict_vertices[(int(point[0]),int(point[1]))]
                                    if (int(point[0]),int(point[1])) in potential_overlap.keys():
                                        old_r, old_g, old_b, old_alpha = potential_overlap[(int(point[0]),int(point[1]))]
                                        # the new ver is on top, correspond to alpha a
                                        alpha /= 255
                                        old_alpha /= 255

                                        new_alpha = alpha + old_alpha * (1-alpha)
                                        new_r = (r*alpha + old_r*old_alpha*(1-alpha))/new_alpha
                                        new_g = (g*alpha + old_g*old_alpha*(1-alpha))/new_alpha
                                        new_b = (b*alpha + old_b*old_alpha*(1-alpha))/new_alpha
                                        potential_overlap[(int(point[0]),int(point[1]))] = (new_r,new_g,new_b,new_alpha*255)
                                        image.im.putpixel((int(point[0]),int(point[1])), (int(new_r), int(new_g), int(new_b), int(new_alpha*255)))
                                        
                                    else:
                                        potential_overlap[(int(point[0]),int(point[1]))] = (r,g,b,alpha)
                                        image.im.putpixel((int(point[0]),int(point[1])), (int(r), int(g), int(b), int(alpha)))


                                 
                elif op == 'trig':
                    points_to_fill_12 = []
                    points_to_fill_13 = []
                    points_to_fill_23 = []

                    color = info[3][1:]

                    if int(info[1]) < 0:
                        point1 = list_vertices[int(info[1])]
                    else:
                        point1 = list_vertices[int(info[1])-1]
                    if int(info[2]) < 0:
                        point2 = list_vertices[int(info[2])]
                    else:
                        point2 = list_vertices[int(info[2])-1]
                    if int(info[3]) < 0:
                        point3 = list_vertices[int(info[3])]
                    else:
                        point3 = list_vertices[int(info[3])-1]
  
                    points_to_fill_12 = create_line_color_step_y(point1, point2)
                    points_to_fill_13 = create_line_color_step_y(point1, point3)
                    points_to_fill_23 = create_line_color_step_y(point2, point3)


                    all_y = list([a[1] for a in points_to_fill_12] +[a[1] for a in points_to_fill_13] + [a[1] for a in points_to_fill_23])
                    all_y = list(set(all_y))
                    all_y.sort()

                    for y in all_y:
                        all_x = list([a[0] for a in points_to_fill_12 if a[1] == y] +[a[0] for a in points_to_fill_13 if a[1] == y] + [a[0] for a in points_to_fill_23 if a[1] == y])
                        all_x = [x for x in all_x if x is not None]
  
                        start_x = min(all_x)
                        end_x = max(all_x)
                        if math.ceil(start_x) < end_x:

                                points_to_fill = create_line_color((start_x, y), (end_x, y))
                                for point in points_to_fill:
                                    r, g, b, alpha = dict_vertices[(int(point[0]),int(point[1]))]
                                    if (int(point[0]),int(point[1])) in potential_overlap.keys():
                                        old_r, old_g, old_b, old_alpha = potential_overlap[(int(point[0]),int(point[1]))]
                                        # the new ver is on top, correspond to alpha a
                                        alpha /= 255
                                        old_alpha /= 255

                                        new_alpha = alpha + old_alpha * (1-alpha)
                                        new_r = (r*alpha + old_r*old_alpha*(1-alpha))/new_alpha
                                        new_g = (g*alpha + old_g*old_alpha*(1-alpha))/new_alpha
                                        new_b = (b*alpha + old_b*old_alpha*(1-alpha))/new_alpha
                                        potential_overlap[(int(point[0]),int(point[1]))] = (new_r,new_g,new_b,new_alpha*255)
                                        image.im.putpixel((int(point[0]),int(point[1])), (int(new_r), int(new_g), int(new_b), int(new_alpha*255)))
                                        
                                    else:
                                        potential_overlap[(int(point[0]),int(point[1]))] = (r,g,b,alpha)
                                        image.im.putpixel((int(point[0]),int(point[1])), (int(r), int(g), int(b), int(alpha)))


                elif op == 'polyec':
                    list_tuples = []
                    points = info[1:-1]
                    color = info[-1][1:]

                    for point_ind in points:
          
                        if int(point_ind) < 0:
                            # extract coordinate info only
                            point = list_vertices[int(point_ind)][0:2]
                        else:
                            point = list_vertices[int(point_ind)-1][0:2]
  
                        list_tuples.append(point)

                    for x in range(1, width):
                        for y in range(1, height):

                            is_in = even_odd_color(x,y,list_tuples)
                            if is_in:
                                image.im.putpixel((x,y), (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), 255))


                elif op == 'polynz':
                    list_tuples = []
                    points = info[1:-1]
                    color = info[-1][1:]
                    dir = {}

                    for i in range(len(points)-1):
                        point1_ind = int(points[i])
                        point2_ind = int(points[i+1])
                        point1 = list_vertices[int(point1_ind)][0:2] if point1_ind < 0 else list_vertices[int(point1_ind)-1][0:2]
                        point2 = list_vertices[int(point2_ind)][0:2] if point2_ind < 0 else list_vertices[int(point2_ind)-1][0:2]
                        
                        if point1[1] < point2[1]:
                            # format: dir[((x1,y1), (x2,y2))]
                            dir[(point1,point2)] = 1
                            
                        else:
                            dir[(point1,point2)] = -1

                    for y in range(1, height):
                        count = 0
                        order_crossing = {}
                        for point_pair in dir.keys():
                            # this y value would cross this line
                            if (point_pair[0][1] > y) != (point_pair[1][1] > y):
                                x_at_y = find_y_on_line(point_pair[0], point_pair[1], y)
                                order_crossing[point_pair] = x_at_y
                        order_crossing = dict(sorted(order_crossing.items(), key = lambda item: item[1]))

                        for x in range(1, width):
                         
                            for tuple_pair in order_crossing.keys():
                                if x < int(order_crossing[tuple_pair]):
                                    if count != 0:
                                        image.im.putpixel((x,y), (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), 255))
                                    break

                                elif x == int(order_crossing[tuple_pair]):
                                    count += dir[tuple_pair] 


                elif op == 'fann':
                    points = info[2:]
                    # a list of tuples, where each tuple consists of the index of the vertices
                    adj = [(points[i], points[i+1]) for i in range(1, len(points)-1)]
                    triangles = [(points[0], triangle[0], triangle[1]) for triangle in adj]
                    for triangle in triangles:
                        point1 = list_vertices[int(triangle[0])-1][0:2]
                        point2 = list_vertices[int(triangle[1])-1][0:2]
                        point3 = list_vertices[int(triangle[2])-1][0:2]

                        points_to_fill_12 = create_line_color_step_y(point1, point2)
                        points_to_fill_13 = create_line_color_step_y(point1, point3)
                        points_to_fill_23 = create_line_color_step_y(point2, point3)

                        all_y = list([a[1] for a in points_to_fill_12] +[a[1] for a in points_to_fill_13] + [a[1] for a in points_to_fill_23])
                       
                        all_y = list(set(all_y))
                        all_y.sort()

                        for y in all_y:
                            
                            all_x = list([a[0] for a in points_to_fill_12 if a[1]==y] +[a[0] for a in points_to_fill_13 if a[1]==y] + [a[0] for a in points_to_fill_23 if a[1]==y])

                            all_x = [x for x in all_x if x is not None]
    
                            start_x = min(all_x)
                            end_x = max(all_x)
                            if math.ceil(start_x) < end_x:

                                points_to_fill = create_line_color((start_x, y), (end_x, y))
                                for point in points_to_fill:
              

                                    r, g, b, alpha = dict_vertices[(int(point[0]),int(point[1]))]
                                    potential_overlap[(int(point[0]),int(point[1]))] = (r,g,b,alpha)
                                    image.im.putpixel((int(point[0]),int(point[1])), (int(r), int(g), int(b), int(alpha)))



                elif op == 'stripn':
                    points = info[2:]
                    # a list of tuples, where each tuple consists of the index of the vertices
                    triangles = [(points[i], points[i+1], points[i+2]) for i in range(0, len(points)-2)]
                    for triangle in triangles:
                        point1 = list_vertices[int(triangle[0])-1][0:2]
                        point2 = list_vertices[int(triangle[1])-1][0:2]
                        point3 = list_vertices[int(triangle[2])-1][0:2]

                        points_to_fill_12 = create_line_color_step_y(point1, point2)
                        points_to_fill_13 = create_line_color_step_y(point1, point3)
                        points_to_fill_23 = create_line_color_step_y(point2, point3)

                        all_y = list([a[1] for a in points_to_fill_12] +[a[1] for a in points_to_fill_13] + [a[1] for a in points_to_fill_23])
                        all_y = list(set(all_y))
                        all_y.sort()

                        for y in all_y:
                            all_x = list([a[0] for a in points_to_fill_12 if a[1] == y] +[a[0] for a in points_to_fill_13 if a[1] == y] + [a[0] for a in points_to_fill_23 if a[1] == y])
                            all_x = [x for x in all_x if x is not None]
    
                            start_x = min(all_x)
                            end_x = max(all_x)
                            if math.ceil(start_x) < end_x:

                                    points_to_fill = create_line_color((start_x, y), (end_x, y))
                                    for point in points_to_fill:
                                        r, g, b, alpha = dict_vertices[(int(point[0]),int(point[1]))]
                                        potential_overlap[(int(point[0]),int(point[1]))] = (r,g,b,alpha)
                                        image.im.putpixel((int(point[0]),int(point[1])), (int(r), int(g), int(b), int(alpha)))                    

                            



            
            image.save(filename)

        # elif mega_info[0] == 'pngs':
        #     prefix = mega_info[3]
        #     count = int(mega_info[4])
        #     print(count)
        #     i = 0

        #     for line in lines[1:]:
        #         image = Image.new("RGBA", (width, height), (0,0,0,0))
        #         filename = str(prefix) + str(str(i).zfill(3)) + '.png'

        #         info = line.split()
        #         op = info[0]
        #         if op == 'xy':
        #             image.im.putpixel((int(info[1]),int(info[2])), (255, 255, 255, 255))
        #         elif op == 'xyrgb':
        #             image.im.putpixel((int(info[1]),int(info[2])), (int(info[3]), int(info[4]), int(info[5]), 255))
        #         elif op == 'xyc':
        #             color = info[3][1:]
        #             image.im.putpixel((int(info[1]),int(info[2])), (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), 255))
        #         image.save(filename)
        #         i += 1
                    
